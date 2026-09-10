package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestAstraReasoningModel(t *testing.T) {
	t.Parallel()

	for _, modelID := range []string{
		"gpt-6-astra", "GPT-6-ASTRA", "gpt-6-astra-2026-09-01", "openai/gpt-6-astra",
	} {
		t.Run(modelID, func(t *testing.T) {
			t.Parallel()
			require.True(t, isReasoningModel(modelID))
			require.True(t, IsResponsesModel(modelID))
			require.True(t, getResponsesModelConfig(modelID).isReasoningModel)
			require.Equal(t, "developer", getResponsesModelConfig(modelID).systemMessageMode)
		})
	}

	for _, modelID := range []string{"gpt-4o", "gpt-4.1", "gpt-6-astral", "gpt-6-other", "not-gpt-6-astra"} {
		t.Run(modelID, func(t *testing.T) {
			t.Parallel()
			require.False(t, isReasoningModel(modelID))
			require.False(t, getResponsesModelConfig(modelID).isReasoningModel)
		})
	}
}

func TestAstraRequestParams(t *testing.T) {
	t.Parallel()

	for _, useResponses := range []bool{false, true} {
		name := "chat"
		if useResponses {
			name = "responses"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			for _, effort := range []ReasoningEffort{ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax} {
				t.Run(string(effort), func(t *testing.T) {
					t.Parallel()
					opts := []Option{WithAPIKey("test-key")}
					if useResponses {
						opts = append(opts, WithUseResponsesAPI())
					}
					provider, err := New(opts...)
					require.NoError(t, err)
					model, err := provider.LanguageModel(t.Context(), "gpt-6-astra")
					require.NoError(t, err)
					call := fantasy.Call{
						Prompt: fantasy.Prompt{
							{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Be helpful."}}},
							{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hello"}}},
						},
						MaxOutputTokens: new(int64(4096)),
						Temperature:     new(0.5),
						TopP:            new(0.9),
					}
					var params any
					if useResponses {
						call.ProviderOptions = NewResponsesProviderOptions(&ResponsesProviderOptions{
							ReasoningEffort:  &effort,
							ReasoningSummary: new("auto"),
						})
						params, _, err = model.(responsesLanguageModel).prepareParams(call)
					} else {
						call.ProviderOptions = NewProviderOptions(&ProviderOptions{
							ReasoningEffort: &effort,
						})
						params, _, err = model.(languageModel).prepareParams(call)
					}
					require.NoError(t, err)
					data, err := json.Marshal(params)
					require.NoError(t, err)
					var body map[string]any
					require.NoError(t, json.Unmarshal(data, &body))
					require.Equal(t, "gpt-6-astra", body["model"])
					require.NotContains(t, body, "max_tokens")
					require.NotContains(t, body, "temperature")
					require.NotContains(t, body, "top_p")
					if useResponses {
						require.Equal(t, float64(4096), body["max_output_tokens"])
						require.Equal(t, map[string]any{"effort": string(effort), "summary": "auto"}, body["reasoning"])
					} else {
						require.Equal(t, float64(4096), body["max_completion_tokens"])
						require.Equal(t, string(effort), body["reasoning_effort"])
					}
				})
			}
		})
	}
}
