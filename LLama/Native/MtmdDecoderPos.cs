using System.Runtime.InteropServices;

namespace LLama.Native;

/// <summary>
/// Decoder position for M-RoPE models.
/// </summary>
/// <remarks>mtmd_decoder_pos</remarks>
[StructLayout(LayoutKind.Sequential)]
public struct MtmdDecoderPos
{
    /// <summary>Temporal position.</summary>
    public uint t;

    /// <summary>Horizontal position.</summary>
    public uint x;

    /// <summary>Vertical position.</summary>
    public uint y;
}
