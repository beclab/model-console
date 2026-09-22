package translate

import "fmt"

// PromptBuilder turns a translation request into a single user message
// for the chat data plane. Inject via Completer.BuildPrompt for
// model-specific templates; nil uses DefaultPrompt.
type PromptBuilder func(from, to Language, text string, autoFrom bool) string

// DefaultPrompt is a generic instruction suitable for most chat-style
// translation models: name the target (and optional source) language in
// English and ask for translation-only output. It is not tied to any
// particular model family.
func DefaultPrompt(from, to Language, text string, autoFrom bool) string {
	if autoFrom || from.Code == "" {
		return fmt.Sprintf(
			"Translate the following text into %s. Note that you should only output the translated result without any additional explanation:\n%s",
			to.NameEN, text)
	}
	return fmt.Sprintf(
		"Translate the following %s text into %s. Note that you should only output the translated result without any additional explanation:\n%s",
		from.NameEN, to.NameEN, text)
}
