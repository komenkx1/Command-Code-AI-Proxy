// Package openai translates OpenAI Chat Completions <-> CommandCode
// native shape so OpenAI SDKs and routers (OpenRouter, LiteLLM,
// langchain.openai) can target this proxy as a drop-in OpenAI base URL.
package openai

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// DefaultProvider is sent as `params.provider` to upstream when the
// caller doesn't specify otherwise.
const DefaultProvider = "command-code"

var stopReasonMap = map[string]string{
	"end_turn":       "stop",
	"stop":           "stop",
	"stop_sequence":  "stop",
	"max_tokens":     "length",
	"length":         "length",
	"tool_use":       "tool_calls",
	"content_filter": "content_filter",
}

// ---------------------------------------------------------------------------
// Tool argument normalisation
// ---------------------------------------------------------------------------
//
// OpenAI-compatible agents often hallucinate parameter names that differ
// from the tool schema they were given (e.g. "file_path" instead of
// "path").  The map below maps *known bad* names to the canonical ones
// we see in real tool schemas.  Only keys that exist in the parsed
// arguments are rewritten; missing keys are left untouched.
//
// The mapping is keyed by the tool function name so we only touch
// arguments for tools we recognise, minimising false positives.

var toolArgNormalizers = map[string]map[string]string{
	"edit": {
		"file_path":     "path",
		"filepath":      "path",
		"filePath":      "path",
		"oldText":       "old_string",
		"old_string":    "old_string",
		"newText":       "new_string",
		"new_string":    "new_string",
	},
	"read": {
		"file_path":     "path",
		"filepath":      "path",
		"filePath":      "path",
	},
	"write": {
		"file_path":     "path",
		"filepath":      "path",
		"filePath":      "path",
	},
	"delete": {
		"file_path":     "path",
		"filepath":      "path",
		"filePath":      "path",
	},
	"bash": {
		"command":       "command",
		"cmd":           "command",
	},
	"search": {
		"search_term":   "query",
		"searchTerm":    "query",
		"term":          "query",
	},
}

// NormalizeToolCallArguments rewrites the JSON-encoded arguments string
// for a known tool so that common hallucinated parameter names are
// replaced with their canonical counterparts.  It returns the (possibly
// unchanged) JSON string and a flag indicating whether any rewrite
// actually happened.
func NormalizeToolCallArguments(toolName, argsJSON string) (string, bool) {
	rules, ok := toolArgNormalizers[toolName]
	if !ok {
		return argsJSON, false
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &parsed); err != nil {
		log.Printf("[toolcall] warning: cannot unmarshal arguments for %q normalization: %v", toolName, err)
		return argsJSON, false
	}

	changed := false
	for badKey, goodKey := range rules {
		if val, exists := parsed[badKey]; exists {
			if _, alreadyOK := parsed[goodKey]; !alreadyOK {
				parsed[goodKey] = val
				delete(parsed, badKey)
				changed = true
				log.Printf("[toolcall] normalized %q → %q for tool %q", badKey, goodKey, toolName)
			}
		}
	}

	if !changed {
		return argsJSON, false
	}

	out, err := json.Marshal(parsed)
	if err != nil {
		log.Printf("[toolcall] warning: cannot marshal normalized arguments for %q: %v", toolName, err)
		return argsJSON, false
	}
	return string(out), true
}

// ChatRequest is the subset of the OpenAI Chat Completions request body
// that we read. Unknown fields are ignored.
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`

	// Optional CommandCode passthrough fields.
	Memory   string         `json:"memory,omitempty"`
	Provider string         `json:"provider,omitempty"`
	Config   map[string]any `json:"config,omitempty"`
}

// ChatMessage is one message in messages[]. Content can be a string or
// an array of content parts; we accept both via the custom unmarshaler
// in normalizeMessages.
type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// CommandCodeBody is the upstream shape this package emits.
type CommandCodeBody struct {
	Memory string         `json:"memory"`
	Params CommandParams  `json:"params"`
	Config map[string]any `json:"config"`
}

// CommandParams is the inner params of the upstream body.
type CommandParams struct {
	Provider string               `json:"provider"`
	Model    string               `json:"model"`
	Messages []CommandCodeMessage `json:"messages"`
}

// CommandCodeMessage is one message in the upstream request.
//
// CommandCode accepts normal chat roles user/assistant. Some clients
// such as 9Router may send OpenAI tool messages without the full
// CommandCode tool-result shape; normalizeMessages flattens those tool
// messages into user text so upstream validation does not reject them.
type CommandCodeMessage struct {
	Role    string                   `json:"role"`
	Content []CommandCodeContentPart `json:"content"`
}

// CommandCodeContentPart is a single typed content fragment. CommandCode
// uses Anthropic-style content blocks; we only emit text parts (image
// inputs are flattened to a placeholder text part since CommandCode
// support for vision through this proxy isn't validated yet).
type CommandCodeContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// FlattenContent collapses a string-or-array content value into a single
// string. Image parts are replaced with a placeholder.
func FlattenContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		out := ""
		for _, p := range c {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if t, ok := m["text"].(string); ok && t != "" {
				out += t
				continue
			}
			if m["type"] == "image_url" {
				out += "[image omitted]"
			}
		}
		return out
	}
	return ""
}

// flattenToParts converts an OpenAI content value (string or array of
// typed parts) into the CommandCode array-of-parts shape.
func flattenToParts(content any) []CommandCodeContentPart {
	switch c := content.(type) {
	case string:
		return []CommandCodeContentPart{{Type: "text", Text: c}}
	case []any:
		parts := make([]CommandCodeContentPart, 0, len(c))
		for _, p := range c {
			switch v := p.(type) {
			case string:
				parts = append(parts, CommandCodeContentPart{Type: "text", Text: v})
			case map[string]any:
				switch v["type"] {
				case "text":
					if t, ok := v["text"].(string); ok {
						parts = append(parts, CommandCodeContentPart{Type: "text", Text: t})
					}
				case "image_url", "image", "input_image":
					parts = append(parts, CommandCodeContentPart{Type: "text", Text: "[image omitted]"})
				}
			}
		}
		if len(parts) == 0 {
			parts = []CommandCodeContentPart{{Type: "text", Text: ""}}
		}
		return parts
	}
	return []CommandCodeContentPart{{Type: "text", Text: ""}}
}

// prependSystemText puts a system instruction text in front of the
// first text part of `parts`, separated by two newlines.
func prependSystemText(sysText string, parts []CommandCodeContentPart) []CommandCodeContentPart {
	if sysText == "" {
		return parts
	}
	if len(parts) == 0 {
		return []CommandCodeContentPart{{Type: "text", Text: sysText}}
	}
	if parts[0].Type == "text" {
		if parts[0].Text == "" {
			parts[0].Text = sysText
		} else {
			parts[0].Text = sysText + "\n\n" + parts[0].Text
		}
		return parts
	}
	return append([]CommandCodeContentPart{{Type: "text", Text: sysText}}, parts...)
}

// normalizeMessages adapts OpenAI messages to the CommandCode shape:
//   - system messages are folded into the next user message (their text
//     is prepended). Multiple consecutive systems are joined with "\n\n".
//     A trailing system (no user after) is attached to the last user
//     message, or otherwise becomes a new user message at the end.
//   - role is restricted to user/assistant. OpenAI tool messages from
//     routers are flattened into user text because CommandCode requires
//     a richer tool-result content shape that those clients do not send.
//   - content is always emitted as a non-empty array of typed parts.
func normalizeMessages(in []ChatMessage) []CommandCodeMessage {
	out := make([]CommandCodeMessage, 0, len(in))
	var pendingSystem string

	appendSystem := func(s string) {
		if pendingSystem == "" {
			pendingSystem = s
		} else {
			pendingSystem += "\n\n" + s
		}
	}

	for _, m := range in {
		role := m.Role
		if role == "" {
			role = "user"
		}
		if role == "system" || role == "developer" {
			appendSystem(FlattenContent(m.Content))
			continue
		}
		if role == "tool" {
			role = "user"
		}
		if role != "user" && role != "assistant" {
			role = "user"
		}
		parts := flattenToParts(m.Content)
		if role == "user" && pendingSystem != "" {
			parts = prependSystemText(pendingSystem, parts)
			pendingSystem = ""
		}
		out = append(out, CommandCodeMessage{Role: role, Content: parts})
	}

	// Trailing system text: attach to last user message, or append a new
	// user message so it isn't silently dropped.
	if pendingSystem != "" {
		attached := false
		for i := len(out) - 1; i >= 0; i-- {
			if out[i].Role == "user" {
				out[i].Content = prependSystemText(pendingSystem, out[i].Content)
				attached = true
				break
			}
		}
		if !attached {
			out = append(out, CommandCodeMessage{
				Role:    "user",
				Content: []CommandCodeContentPart{{Type: "text", Text: pendingSystem}},
			})
		}
	}

	return out
}

// BuildDefaultConfig is used when the OpenAI request didn't pass a
// `config` field. The values mirror what the CommandCode CLI sends.
func BuildDefaultConfig() map[string]any {
	return map[string]any{
		"workingDir":    "",
		"date":          time.Now().UTC().Format("2006-01-02"),
		"environment":   fmt.Sprintf("%s-%s, Go %s", runtime.GOOS, runtime.GOARCH, runtime.Version()),
		"structure":     []string{},
		"isGitRepo":     false,
		"currentBranch": "",
		"mainBranch":    "main",
		"gitStatus":     "",
		"recentCommits": []string{},
	}
}

// ToCommandCode validates the OpenAI request and returns the upstream
// body. Returns an error with an OpenAI-style message string.
func ToCommandCode(req ChatRequest) (CommandCodeBody, error) {
	if req.Model == "" {
		return CommandCodeBody{}, errors.New(`field "model" is required`)
	}
	if len(req.Messages) == 0 {
		return CommandCodeBody{}, errors.New(`field "messages" must be a non-empty array`)
	}
	provider := req.Provider
	if provider == "" {
		provider = DefaultProvider
	}
	cfg := req.Config
	if cfg == nil {
		cfg = BuildDefaultConfig()
	}
	return CommandCodeBody{
		Memory: req.Memory,
		Params: CommandParams{
			Provider: provider,
			Model:    req.Model,
			Messages: normalizeMessages(req.Messages),
		},
		Config: cfg,
	}, nil
}

// MapFinishReason translates upstream stop_reason to OpenAI finish_reason.
func MapFinishReason(s string) string {
	if s == "" {
		return "stop"
	}
	if v, ok := stopReasonMap[s]; ok {
		return v
	}
	return "stop"
}

// FlattenUpstreamContent extracts text from the upstream content shape,
// which is an array of {type:"text", text:"..."} objects.
func FlattenUpstreamContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		out := ""
		for _, p := range c {
			switch v := p.(type) {
			case string:
				out += v
			case map[string]any:
				if t, ok := v["text"].(string); ok {
					out += t
				}
			}
		}
		return out
	}
	return ""
}

// ChatCompletion is the OpenAI Chat Completion response shape.
type ChatCompletion struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []ChatChoice   `json:"choices"`
	Usage   ChatUsage      `json:"usage"`
}

// ChatChoice is one entry in choices[].
type ChatChoice struct {
	Index        int            `json:"index"`
	Message      ChatRespMsg    `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

// ChatRespMsg is the response message body.
// Content is a pointer so it serializes as JSON null when the response
// contains only tool_calls (OpenAI spec requires content:null in that case).
type ChatRespMsg struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall is one tool invocation in the OpenAI response.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the function details within a ToolCall.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded string
}

// StreamToolCallDelta is one tool call delta in a streaming chunk.
type StreamToolCallDelta struct {
	Index    int                  `json:"index"`
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function *StreamFunctionDelta `json:"function,omitempty"`
}

// StreamFunctionDelta is the function portion of a streaming tool call delta.
type StreamFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ChatUsage is OpenAI's usage shape.
type ChatUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// ---------------------------------------------------------------------------
// Tool-call extraction from upstream Anthropic-style content blocks
// ---------------------------------------------------------------------------

// textToolCallRe matches the text-based tool call pattern that some models
// emit instead of structured tool_use blocks. The pattern looks like:
//
//	**Calling:** `function_name`
//	```
//	{"arg": "value"}
//	```
//
// We also handle the variant with optional json/JSON language tag after ```.
var textToolCallRe = regexp.MustCompile(
	"(?s)" + // dot matches newline
		`\*\*Calling:\*\*\s*` + "`" + `([^` + "`" + `]+)` + "`" + `\s*\n` +
		"```" + `(?:json|JSON)?\s*\n(.*?)\n` + "```",
)

// ExtractToolUseBlocks extracts native Anthropic tool_use content blocks
// from the upstream content array and returns them as OpenAI ToolCalls.
// Non-tool_use blocks are ignored. Returns nil if no tool_use blocks found.
func ExtractToolUseBlocks(content any) []ToolCall {
	arr, ok := content.([]any)
	if !ok {
		return nil
	}
	var calls []ToolCall
	for _, block := range arr {
		m, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] != "tool_use" {
			continue
		}
		name, _ := m["name"].(string)
		id, _ := m["id"].(string)
		if id == "" {
			id = "call_" + randomHex(12)
		}
		// Marshal input to JSON string (OpenAI expects arguments as string)
		var args string
		if input := m["input"]; input != nil {
			b, err := json.Marshal(input)
			if err != nil {
				log.Printf("[toolcall] warning: failed to marshal tool_use input for %q: %v", name, err)
				args = "{}"
			} else {
				args = string(b)
			}
		} else {
			args = "{}"
		}
		// Normalize common hallucinated parameter names
		if norm, ok := NormalizeToolCallArguments(name, args); ok {
			args = norm
		}
		log.Printf("[toolcall] extracted native tool_use block: name=%q id=%q args_len=%d", name, id, len(args))
		calls = append(calls, ToolCall{
			ID:   id,
			Type: "function",
			Function: ToolCallFunction{
				Name:      name,
				Arguments: args,
			},
		})
	}
	return calls
}

// ExtractTextContent extracts only the text from text-type content blocks,
// skipping tool_use blocks entirely. This is used instead of
// FlattenUpstreamContent when we need to separate text from tool calls.
func ExtractTextContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var sb strings.Builder
		for _, p := range c {
			switch v := p.(type) {
			case string:
				sb.WriteString(v)
			case map[string]any:
				// Only extract text blocks, skip tool_use and others
				if v["type"] == "text" {
					if t, ok := v["text"].(string); ok {
						sb.WriteString(t)
					}
				}
			}
		}
		return sb.String()
	}
	return ""
}

// ParseTextToolCalls detects tool calls embedded as formatted text in the
// response content. Some models (e.g. Kimi-K2.5 via CommandCode) emit
// tool calls as markdown text like:
//
//	**Calling:** `read`
//	```
//	{"file_path": "/path/to/file"}
//	```
//
// Returns the parsed ToolCalls and the remaining text with tool call
// blocks removed. Returns nil calls if no pattern matched.
func ParseTextToolCalls(text string) ([]ToolCall, string) {
	matches := textToolCallRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return nil, text
	}

	var calls []ToolCall
	for _, loc := range matches {
		// loc[2:4] = function name capture, loc[4:6] = arguments capture
		name := text[loc[2]:loc[3]]
		rawArgs := strings.TrimSpace(text[loc[4]:loc[5]])

		// Validate that arguments look like JSON
		if !json.Valid([]byte(rawArgs)) {
			log.Printf("[toolcall] warning: text-based tool call %q has invalid JSON arguments, wrapping as string: %s", name, rawArgs)
			// Wrap non-JSON arguments as a JSON string value
			b, _ := json.Marshal(map[string]string{"raw": rawArgs})
			rawArgs = string(b)
		}

		// Normalize common hallucinated parameter names
		if norm, ok := NormalizeToolCallArguments(name, rawArgs); ok {
			rawArgs = norm
		}

		id := "call_" + randomHex(12)
		log.Printf("[toolcall] parsed text-based tool call: name=%q id=%q args_len=%d", name, id, len(rawArgs))
		calls = append(calls, ToolCall{
			ID:   id,
			Type: "function",
			Function: ToolCallFunction{
				Name:      name,
				Arguments: rawArgs,
			},
		})
	}

	// Remove matched tool call text from content
	remaining := text
	// Process in reverse order to preserve indices
	for i := len(matches) - 1; i >= 0; i-- {
		loc := matches[i]
		remaining = remaining[:loc[0]] + remaining[loc[1]:]
	}
	remaining = strings.TrimSpace(remaining)

	log.Printf("[toolcall] text-based detection: found %d tool call(s), remaining_text_len=%d", len(calls), len(remaining))
	return calls, remaining
}

// strPtr returns a pointer to s. Used to populate ChatRespMsg.Content.
func strPtr(s string) *string { return &s }

// ToOpenAI converts an upstream parsed body into an OpenAI ChatCompletion.
// It handles three content scenarios:
//  1. Plain text only → content is the text, no tool_calls.
//  2. Native tool_use blocks (Anthropic-style) → tool_calls populated,
//     content is null or the accompanying text.
//  3. Text-based tool calls (model emits "**Calling:** `fn`\n```\n{}\n```")
//     → parsed into tool_calls, remaining text kept in content.
func ToOpenAI(upstream map[string]any, requestModel string) ChatCompletion {
	id, _ := upstream["id"].(string)
	if id == "" {
		id = "chatcmpl-" + randomHex(12)
	}
	role, _ := upstream["role"].(string)
	if role == "" {
		role = "assistant"
	}
	stopReason, _ := upstream["stop_reason"].(string)

	model := requestModel
	if model == "" {
		if m, ok := upstream["model"].(string); ok {
			model = m
		}
	}

	usage, _ := upstream["usage"].(map[string]any)
	in := numAsInt(usage["input_tokens"])
	out := numAsInt(usage["output_tokens"])

	rawContent := upstream["content"]

	// Step 1: Try extracting native Anthropic tool_use blocks
	toolCalls := ExtractToolUseBlocks(rawContent)
	textContent := ExtractTextContent(rawContent)

	// Step 2: If no native tool_use blocks, try parsing text-based tool calls
	if len(toolCalls) == 0 && textContent != "" {
		parsed, remaining := ParseTextToolCalls(textContent)
		if len(parsed) > 0 {
			toolCalls = parsed
			textContent = remaining
			// Override stop_reason since we detected tool calls in text
			log.Printf("[toolcall] overriding stop_reason from %q to \"tool_use\" (detected %d text-based tool call(s))", stopReason, len(parsed))
			stopReason = "tool_use"
		}
	}

	// Step 3: Build the response message
	var msg ChatRespMsg
	if len(toolCalls) > 0 {
		log.Printf("[toolcall] response contains %d tool call(s), finish_reason=tool_calls", len(toolCalls))
		if textContent == "" {
			msg = ChatRespMsg{Role: role, Content: nil, ToolCalls: toolCalls}
		} else {
			msg = ChatRespMsg{Role: role, Content: strPtr(textContent), ToolCalls: toolCalls}
		}
	} else {
		msg = ChatRespMsg{Role: role, Content: strPtr(textContent)}
	}

	return ChatCompletion{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []ChatChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: MapFinishReason(stopReason),
			},
		},
		Usage: ChatUsage{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out},
	}
}

// StreamChunk is one frame of the SSE response.
type StreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []StreamChoice      `json:"choices"`
}

// StreamChoice is one choice within a chunk.
type StreamChoice struct {
	Index        int                    `json:"index"`
	Delta        map[string]any         `json:"delta"`
	FinishReason any                    `json:"finish_reason"`
}

// BuildStreamChunks emits the role chunk -> content chunk -> finish
// chunk sequence for an already-buffered upstream response. When tool
// calls are detected (native or text-based), it emits tool_calls deltas
// instead of (or in addition to) content deltas. Output is terminated
// externally by writing "data: [DONE]\n\n".
func BuildStreamChunks(upstream map[string]any, requestModel string) []StreamChunk {
	id, _ := upstream["id"].(string)
	if id == "" {
		id = "chatcmpl-" + randomHex(12)
	}
	model := requestModel
	if model == "" {
		if m, ok := upstream["model"].(string); ok {
			model = m
		}
	}
	created := time.Now().Unix()
	stopReason, _ := upstream["stop_reason"].(string)

	rawContent := upstream["content"]

	// Extract tool calls (same logic as ToOpenAI)
	toolCalls := ExtractToolUseBlocks(rawContent)
	textContent := ExtractTextContent(rawContent)

	if len(toolCalls) == 0 && textContent != "" {
		parsed, remaining := ParseTextToolCalls(textContent)
		if len(parsed) > 0 {
			toolCalls = parsed
			textContent = remaining
			stopReason = "tool_use"
		}
	}

	finishReason := MapFinishReason(stopReason)

	// Chunk 1: role
	chunks := []StreamChunk{
		{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []StreamChoice{{Index: 0, Delta: map[string]any{"role": "assistant"}, FinishReason: nil}},
		},
	}

	// Chunk 2+: content text (if any)
	if textContent != "" {
		chunks = append(chunks, StreamChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []StreamChoice{{Index: 0, Delta: map[string]any{"content": textContent}, FinishReason: nil}},
		})
	}

	// Chunk 3+: tool call deltas (if any)
	if len(toolCalls) > 0 {
		for i, tc := range toolCalls {
			// First chunk for this tool call: id + name
			chunks = append(chunks, StreamChunk{
				ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []StreamChoice{{
					Index: 0,
					Delta: map[string]any{
						"tool_calls": []StreamToolCallDelta{{
							Index: i,
							ID:    tc.ID,
							Type:  "function",
							Function: &StreamFunctionDelta{
								Name:      tc.Function.Name,
								Arguments: "",
							},
						}},
					},
					FinishReason: nil,
				}},
			})
			// Second chunk: arguments
			chunks = append(chunks, StreamChunk{
				ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []StreamChoice{{
					Index: 0,
					Delta: map[string]any{
						"tool_calls": []StreamToolCallDelta{{
							Index: i,
							Function: &StreamFunctionDelta{
								Arguments: tc.Function.Arguments,
							},
						}},
					},
					FinishReason: nil,
				}},
			})
		}
	}

	// Final chunk: finish_reason
	chunks = append(chunks, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []StreamChoice{{Index: 0, Delta: map[string]any{}, FinishReason: finishReason}},
	})
	return chunks
}

// Model is one entry in the /v1/models response.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the /v1/models payload.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// BuildModelsResponse returns the OpenAI-shaped list for the given IDs.
func BuildModelsResponse(ids []string) ModelsResponse {
	now := time.Now().Unix()
	out := make([]Model, 0, len(ids))
	for _, id := range ids {
		out = append(out, Model{ID: id, Object: "model", Created: now, OwnedBy: DefaultProvider})
	}
	return ModelsResponse{Object: "list", Data: out}
}

// MergeModelLists deduplicates model IDs across multiple sources while
// preserving order: env-configured IDs first (so the static
// OPENAI_MODELS list stays the canonical advertised set), then any
// extras (e.g. models discovered from analytics history) appended in
// order. Empty strings are skipped.
func MergeModelLists(sources ...[]string) []string {
	seen := make(map[string]struct{}, 16)
	var out []string
	for _, src := range sources {
		for _, id := range src {
			if id == "" {
				continue
			}
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			out = append(out, id)
		}
	}
	return out
}

func numAsInt(v any) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case int64:
		return t
	}
	return 0
}

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// astronomically unlikely; fall back to deterministic value
		return "abcdef0123456789"[:n*2]
	}
	return hex.EncodeToString(b)
}
