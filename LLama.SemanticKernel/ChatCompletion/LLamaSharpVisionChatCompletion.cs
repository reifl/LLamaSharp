#if NET6_0_OR_GREATER
using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Native;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using static LLama.InteractiveExecutor;
using static LLama.LLamaTransforms;
using SKChatHistory  = Microsoft.SemanticKernel.ChatCompletion.ChatHistory;
using SKAuthorRole   = Microsoft.SemanticKernel.ChatCompletion.AuthorRole;

namespace LLamaSharp.SemanticKernel.ChatCompletion;

/// <summary>
/// Vision-capable LLamaSharp ChatCompletion service for Semantic Kernel.
/// Handles <see cref="ImageContent"/> items in chat messages by loading them
/// into <see cref="InteractiveExecutor.Embeds"/> and inserting the model's
/// media marker at the corresponding position in the text prompt.
/// Uses the Jinja2 chat template via <c>llama-jinja.dll</c> when available,
/// falling back to the built-in heuristic formatter.
/// </summary>
/// <remarks>
/// The <see cref="InteractiveExecutor"/> passed to the constructor must have been
/// created with a <see cref="MtmdWeights"/> argument (i.e.
/// <see cref="InteractiveExecutor.IsMultiModal"/> must be <see langword="true"/>).
/// </remarks>
public sealed class LLamaSharpVisionChatCompletion : IChatCompletionService
{
    // ── Fields ───────────────────────────────────────────────────────────────
    private readonly InteractiveExecutor _executor;
    private readonly LLamaWeights        _model;
    private readonly LLamaSharpPromptExecutionSettings _defaultSettings;
    private readonly ITextStreamTransform _outputTransform;
    private readonly HistoryTransform     _historyTransform = new();
    private readonly string               _mediaMarker;
    private readonly bool                 _isStateful;

    private readonly Dictionary<string, object?> _attributes = new();
    public IReadOnlyDictionary<string, object?> Attributes => _attributes;

    // ── Constructor ──────────────────────────────────────────────────────────
    /// <param name="executor">
    ///   A multimodal <see cref="InteractiveExecutor"/> created with <see cref="MtmdWeights"/>.
    /// </param>
    /// <param name="model">The loaded model weights — needed for Jinja2 template formatting.</param>
    /// <param name="defaultSettings">Optional default inference settings.</param>
    /// <param name="outputTransform">Optional output post-processor.</param>
    /// <exception cref="ArgumentException">
    ///   Thrown when <paramref name="executor"/> has no multimodal model loaded.
    /// </exception>
    public LLamaSharpVisionChatCompletion(
        InteractiveExecutor executor,
        LLamaWeights model,
        LLamaSharpPromptExecutionSettings? defaultSettings = null,
        ITextStreamTransform? outputTransform = null)
    {
        if (!executor.IsMultiModal)
            throw new ArgumentException(
                "The executor must be created with MtmdWeights. " +
                "Use: new InteractiveExecutor(context, mtmdWeights)",
                nameof(executor));

        _executor   = executor;
        _model      = model;
        _isStateful = executor is StatefulExecutorBase;

        _defaultSettings = defaultSettings ?? new LLamaSharpPromptExecutionSettings
        {
            MaxTokens     = 512,
            Temperature   = 0.7,
            TopP          = 0.9,
            StopSequences = new List<string>(),
        };
        _outputTransform = outputTransform ?? new KeywordTextOutputStreamTransform(new[]
        {
            $"{LLama.Common.AuthorRole.User}:",
            $"{LLama.Common.AuthorRole.Assistant}:",
            $"{LLama.Common.AuthorRole.System}:",
        });
        _mediaMarker = NativeApi.MtmdDefaultMarker() ?? "<image>";
    }

    // ── IChatCompletionService ───────────────────────────────────────────────

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        SKChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var settings   = ResolveSettings(executionSettings);
        var inferParams = settings.ToLLamaSharpInferenceParams();
        var prompt     = PrepareRequest(chatHistory);

        var result = _executor.InferAsync(prompt, inferParams, cancellationToken);
        var output = _outputTransform.TransformAsync(result);

        var sb = new StringBuilder();
        await foreach (var token in output.WithCancellation(cancellationToken))
            sb.Append(token);

        // Strip anti-prompt suffix included in the last token
        string text = StripAntiPromptSuffix(sb.ToString(), inferParams.AntiPrompts);

        return new List<ChatMessageContent>
        {
            new(SKAuthorRole.Assistant, text)
        }.AsReadOnly();
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        SKChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var settings    = ResolveSettings(executionSettings);
        var inferParams = settings.ToLLamaSharpInferenceParams();
        var prompt      = PrepareRequest(chatHistory);

        var result = _executor.InferAsync(prompt, inferParams, cancellationToken);
        var output = _outputTransform.TransformAsync(result);

        // Lookahead buffer: hold back the last N chars so we can strip any
        // anti-prompt suffix that InferAsync emits as the final token(s).
        var antiPrompts = inferParams.AntiPrompts.ToList();
        int holdLen     = antiPrompts.Count > 0 ? antiPrompts.Max(a => a.Length) : 0;
        var buffer      = new StringBuilder();

        await foreach (var token in output.WithCancellation(cancellationToken))
        {
            buffer.Append(token);

            if (buffer.Length > holdLen)
            {
                string safe = buffer.ToString(0, buffer.Length - holdLen);
                yield return new StreamingChatMessageContent(SKAuthorRole.Assistant, safe);
                buffer.Remove(0, safe.Length);
            }
        }

        // Flush remainder, stripping any trailing anti-prompt
        string tail = StripAntiPromptSuffix(buffer.ToString(), antiPrompts);
        tail = tail.TrimEnd();
        if (tail.Length > 0)
            yield return new StreamingChatMessageContent(SKAuthorRole.Assistant, tail);
    }

    // ── Prompt preparation ───────────────────────────────────────────────────

    /// <summary>
    /// Builds the LLama ChatHistory with media markers, loads image embeds,
    /// and formats the prompt delta via Jinja2 (or heuristic fallback).
    /// </summary>
    private string PrepareRequest(SKChatHistory chatHistory)
    {
        _executor.Embeds.Clear();

        // Convert full SK history → LLama history (images → media markers)
        var llamaHistory = BuildLlamaHistory(chatHistory);

        return FormatPromptDelta(llamaHistory);
    }

    /// <summary>
    /// Converts <paramref name="skHistory"/> into a <see cref="LLama.Common.ChatHistory"/>,
    /// replacing <see cref="ImageContent"/> items with the media marker and loading them
    /// into <see cref="InteractiveExecutor.Embeds"/> (only for the last user message).
    /// </summary>
    private LLama.Common.ChatHistory BuildLlamaHistory(SKChatHistory skHistory)
    {
        var llamaHistory  = new LLama.Common.ChatHistory();
        var messages      = skHistory.ToList();
        int lastUserIndex = -1;

        // Find the last user message — only those images go into Embeds this turn
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == SKAuthorRole.User) { lastUserIndex = i; break; }
        }

        for (int i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            if (!Enum.TryParse<LLama.Common.AuthorRole>(
                    message.Role.Label, ignoreCase: true, out var role))
                role = LLama.Common.AuthorRole.Unknown;

            var sb = new StringBuilder();

            if (message.Items is { Count: > 0 })
            {
                foreach (var item in message.Items)
                {
                    switch (item)
                    {
                        case TextContent text:
                            sb.Append(text.Text);
                            break;

                        case ImageContent image:
                            if (i == lastUserIndex)
                            {
                                // Load embed — only for the new (last) user message
                                _executor.Embeds.Add(LoadEmbed(image));
                            }
                            sb.Append(_mediaMarker);
                            break;
                    }
                }
            }
            else if (!string.IsNullOrEmpty(message.Content))
            {
                sb.Append(message.Content);
            }

            llamaHistory.AddMessage(role, sb.ToString());
        }

        return llamaHistory;
    }

    /// <summary>
    /// Formats the prompt delta to feed into the (stateful) executor.
    /// Uses Jinja2 where available; falls back to <see cref="HistoryTransform"/>.
    /// For stateful executors after the first run, only the new message delta is returned.
    /// </summary>
    private string FormatPromptDelta(LLama.Common.ChatHistory history)
    {
        if (!history.Messages.Any())
            return string.Empty;

        // Check if this is the first inference run (executor has no prior context)
        bool isFirstRun = true;
        if (_isStateful)
        {
            var state = (InteractiveExecutorState)((StatefulExecutorBase)_executor).GetStateData();
            isFirstRun = state.IsPromptRun;
        }

        if (isFirstRun)
        {
            // First run: send the entire formatted history
            return FormatHistory(history, addAssistant: true);
        }
        else
        {
            // Subsequent runs: send only the delta (new user turn + assistant prompt)
            // past = history without the last message
            var pastHistory = new LLama.Common.ChatHistory();
            foreach (var msg in history.Messages.Take(history.Messages.Count - 1))
                pastHistory.AddMessage(msg.AuthorRole, msg.Content);

            string past = FormatHistory(pastHistory, addAssistant: false);
            string full = FormatHistory(history,     addAssistant: true);

            if (!full.StartsWith(past, StringComparison.Ordinal))
                return full; // Template changed shape — return full to be safe

            string delta = full[past.Length..];

            // Preserve leading newline if needed
            if (past.Length > 0 && past.EndsWith('\n') && delta.Length > 0 && delta[0] != '\n')
                delta = "\n" + delta;

            return delta;
        }
    }

    /// <summary>
    /// Formats a <see cref="LLama.Common.ChatHistory"/> using Jinja2 via
    /// <c>llama-jinja.dll</c>. Falls back to <see cref="HistoryTransform"/> if
    /// Jinja2 is unavailable or returns an error.
    /// </summary>
    private string FormatHistory(LLama.Common.ChatHistory history, bool addAssistant)
    {
        if (!history.Messages.Any())
            return string.Empty;

        var msgs = history.Messages.Select(
            m => (m.AuthorRole.ToString().ToLowerInvariant(), m.Content));

        string? rendered = LLamaJinjaTemplate.ApplyTemplate(
            _model, msgs, addGenerationPrompt: addAssistant);

        return rendered ?? _historyTransform.HistoryToText(history);
    }

    // ── Image loading ────────────────────────────────────────────────────────

    /// <summary>
    /// Loads a single <see cref="ImageContent"/> via ImageSharp (JPEG, PNG, WebP, …)
    /// into a <see cref="SafeMtmdEmbed"/>.
    /// </summary>
    private static SafeMtmdEmbed LoadEmbed(ImageContent image)
    {
        Stream stream;
        if (image.Data.HasValue && !image.Data.Value.IsEmpty)
        {
            stream = new MemoryStream(image.Data.Value.ToArray());
        }
        else if (image.Uri is not null)
        {
            if (!image.Uri.IsFile)
                throw new NotSupportedException(
                    $"Remote image URIs are not supported. Got: {image.Uri}");
            stream = File.OpenRead(image.Uri.LocalPath);
        }
        else
        {
            throw new ArgumentException(
                "ImageContent must have either Data or a local file URI.", nameof(image));
        }

        using (stream)
        using (var img = SixLabors.ImageSharp.Image.Load<Rgb24>(stream))
        {
            int w = img.Width, h = img.Height;
            var rgb = new byte[w * h * 3];
            img.CopyPixelDataTo(rgb.AsSpan());
            return SafeMtmdEmbed.FromRgbBytes((uint)w, (uint)h, rgb)
                ?? throw new InvalidOperationException("mtmd_bitmap_init failed.");
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private LLamaSharpPromptExecutionSettings ResolveSettings(PromptExecutionSettings? settings) =>
        settings != null
            ? LLamaSharpPromptExecutionSettings.FromRequestSettings(settings)
            : _defaultSettings;

    private static string StripAntiPromptSuffix(string text, IEnumerable<string> antiPrompts)
    {
        foreach (var anti in antiPrompts)
        {
            if (text.EndsWith(anti, StringComparison.Ordinal))
                return text[..^anti.Length].TrimEnd();
        }
        return text.TrimEnd();
    }
}
#endif
