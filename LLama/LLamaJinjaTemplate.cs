using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace LLama;

/// <summary>
/// Renders a chat history to a prompt string using the Jinja2 template embedded
/// in the GGUF model, via <c>llama-jinja.dll</c>.
/// </summary>
/// <remarks>
/// The <c>llama-jinja.dll</c> native library must be loaded before calling
/// <see cref="ApplyTemplate"/>. If it is not loaded (or if the call fails),
/// <see cref="ApplyTemplate"/> returns <see langword="null"/> and the caller
/// should fall back to a heuristic formatter.
/// </remarks>
public static class LLamaJinjaTemplate
{
    // ── P/Invoke ─────────────────────────────────────────────────────────────

    private const string JinjaDll = "llama-jinja";

    /// <summary>
    /// messagesJson is a UTF-8 encoded, null-terminated C string (passed as IntPtr).
    /// buf may be null (pass 0 for bufSize) to query the required length.
    /// Returns the number of bytes written (excluding null terminator), or a
    /// negative value on error.
    /// </summary>
    [DllImport(JinjaDll, CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int llama_jinja_apply_template(
        IntPtr model,
        IntPtr messagesJson,
        [MarshalAs(UnmanagedType.I1)] bool addGenerationPrompt,
        [MarshalAs(UnmanagedType.I1)] bool enableThinking,
        byte* buf,
        int bufSize);

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Applies the Jinja2 chat template embedded in <paramref name="model"/> to
    /// the given list of messages.
    /// </summary>
    /// <param name="model">The loaded model weights.</param>
    /// <param name="messages">
    ///   Sequence of <c>(role, content)</c> pairs in conversation order.
    ///   Roles should be lowercase strings such as <c>"system"</c>,
    ///   <c>"user"</c>, and <c>"assistant"</c>.
    /// </param>
    /// <param name="addGenerationPrompt">
    ///   When <see langword="true"/> the template appends the model's
    ///   generation-start marker (e.g. <c>&lt;start_of_turn&gt;model\n</c>).
    /// </param>
    /// <param name="enableThinking">
    ///   Enables the thinking / chain-of-thought channel for models that
    ///   support it (e.g. Gemma 4).
    /// </param>
    /// <returns>
    ///   The rendered prompt string, or <see langword="null"/> if
    ///   <c>llama-jinja.dll</c> is not loaded or the template call fails.
    /// </returns>
    public static unsafe string? ApplyTemplate(
        LLamaWeights model,
        IEnumerable<(string role, string content)> messages,
        bool addGenerationPrompt = true,
        bool enableThinking = false)
    {
        try
        {
            var msgObjects = new List<object>();
            foreach (var (role, content) in messages)
                msgObjects.Add(new { role, content });

            // Encode JSON to UTF-8 bytes with null terminator for the native call
            byte[] jsonBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(msgObjects) + "\0");
            IntPtr modelPtr = model.NativeHandle.DangerousGetHandle();

            fixed (byte* jsonPtr = jsonBytes)
            {
                int needed = llama_jinja_apply_template(
                    modelPtr, (IntPtr)jsonPtr, addGenerationPrompt, enableThinking, null, 0);

                if (needed < 0)
                    return null;

                var buf = new byte[needed + 1];
                fixed (byte* p = buf)
                    llama_jinja_apply_template(
                        modelPtr, (IntPtr)jsonPtr, addGenerationPrompt, enableThinking, p, needed + 1);

                return Encoding.UTF8.GetString(buf, 0, needed);
            }
        }
        catch
        {
            // llama-jinja.dll not loaded or call failed
            return null;
        }
    }
}
