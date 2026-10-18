// wincap: livetrans 서버가 쓰는 Windows 오디오 캡처 헬퍼.
//   wincap list                 보이는 최상위 창 목록을 JSON 으로 stdout 에 출력
//   wincap capture system       기본 출력 장치 소리 전체 (WASAPI loopback)
//   wincap capture pid <PID>    해당 프로세스 트리의 소리만 (Windows 10 2004+ process loopback)
// capture 는 16 kHz mono PCM16 LE 를 stdout 으로 계속 내보낸다. stdin 이 닫히면 종료.
// 소리가 안 나는 동안 WASAPI 가 패킷을 안 주므로, 벽시계 기준으로 모자란 만큼 무음을 채워 시간축(t_cap)을 유지한다.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using NAudio.CoreAudioApi;
using NAudio.CoreAudioApi.Interfaces;
using NAudio.Wasapi.CoreAudioApi.Interfaces;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

const int OutRate = 16000;

if (args.Length >= 1 && args[0] == "list")
{
    Console.OutputEncoding = new UTF8Encoding(false);
    Console.WriteLine(JsonSerializer.Serialize(Windows.List()));
    return 0;
}
if (args.Length >= 2 && args[0] == "capture")
{
    try
    {
        Process? target = null;
        IWaveIn capture;
        if (args[1] == "system")
        {
            capture = new WasapiLoopbackCapture();
        }
        else if (args[1] == "pid" && args.Length >= 3 && int.TryParse(args[2], out var pid))
        {
            // process loopback 은 없는 PID 로도 활성화에 성공하므로 직접 확인한다
            try { target = Process.GetProcessById(pid); }
            catch (ArgumentException) { Console.Error.WriteLine($"ERROR target process {pid} not found"); return 4; }
            capture = await ProcessLoopbackCapture.CreateAsync(pid);
        }
        else
        {
            throw new ArgumentException("usage: capture system | capture pid <PID>");
        }
        return Run(capture, target);
    }
    catch (Exception e)
    {
        Console.Error.WriteLine($"ERROR {e.GetType().Name}: {e.Message}");
        return 2;
    }
}
Console.Error.WriteLine("usage: wincap list | wincap capture system | wincap capture pid <PID>");
return 1;

static int Run(IWaveIn capture, Process? target)
{
    var buffer = new BufferedWaveProvider(capture.WaveFormat)
    {
        DiscardOnBufferOverflow = true,
        BufferDuration = TimeSpan.FromSeconds(5),
        ReadFully = false,
    };
    ISampleProvider sp = buffer.ToSampleProvider();
    if (sp.WaveFormat.Channels > 1) sp = new Downmix(sp);
    if (sp.WaveFormat.SampleRate != OutRate) sp = new WdlResamplingSampleProvider(sp, OutRate);
    var pcm16 = new SampleToWaveProvider16(sp);

    Exception? failure = null;
    var stopped = new ManualResetEventSlim(false);
    capture.DataAvailable += (_, e) => buffer.AddSamples(e.Buffer, 0, e.BytesRecorded);
    capture.RecordingStopped += (_, e) => { failure = e.Exception; stopped.Set(); };

    // stdin 이 닫히면(서버가 종료) 끝낸다
    var quit = new ManualResetEventSlim(false);
    new Thread(() => { try { while (Console.In.Read() >= 0) { } } catch { } quit.Set(); }) { IsBackground = true }.Start();

    var stdout = Console.OpenStandardOutput();
    capture.StartRecording();
    Console.Error.WriteLine($"READY in={capture.WaveFormat} out={OutRate}Hz mono s16le");
    Console.Error.Flush();

    var clock = Stopwatch.StartNew();
    long emitted = 0;  // 내보낸 샘플 수
    var chunk = new byte[OutRate / 50 * 2 * 4];  // 80 ms
    var zeros = new byte[chunk.Length];
    try
    {
        while (!quit.IsSet && !stopped.IsSet)
        {
            if (target is { HasExited: true })
            {
                Console.Error.WriteLine("ERROR target process exited");
                capture.StopRecording();
                return 5;
            }
            int n;
            while ((n = pcm16.Read(chunk, 0, chunk.Length)) > 0)
            {
                stdout.Write(chunk, 0, n);
                emitted += n / 2;
            }
            // 무음 구간: 벽시계보다 100 ms 넘게 뒤처지면 0 으로 채운다
            long expected = clock.ElapsedMilliseconds * OutRate / 1000;
            while (expected - emitted > OutRate / 10)
            {
                int fill = (int)Math.Min(zeros.Length / 2, expected - emitted);
                stdout.Write(zeros, 0, fill * 2);
                emitted += fill;
            }
            stdout.Flush();
            Thread.Sleep(20);
        }
    }
    catch (IOException)
    {
        // 서버가 파이프를 닫음
    }
    capture.StopRecording();
    if (failure != null)
    {
        Console.Error.WriteLine($"ERROR capture stopped: {failure.Message}");
        return 3;
    }
    return 0;
}

sealed class Downmix(ISampleProvider source) : ISampleProvider
{
    readonly int _ch = source.WaveFormat.Channels;
    float[] _buf = [];
    public WaveFormat WaveFormat { get; } = WaveFormat.CreateIeeeFloatWaveFormat(source.WaveFormat.SampleRate, 1);

    public int Read(float[] buffer, int offset, int count)
    {
        int need = count * _ch;
        if (_buf.Length < need) _buf = new float[need];
        int got = source.Read(_buf, 0, need) / _ch;
        for (int i = 0; i < got; i++)
        {
            float s = 0;
            for (int c = 0; c < _ch; c++) s += _buf[i * _ch + c];
            buffer[offset + i] = s / _ch;
        }
        return got;
    }
}

static class Windows
{
    delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr hWnd, StringBuilder s, int max);
    [DllImport("user32.dll")] static extern int GetWindowTextLength(IntPtr hWnd);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")] static extern IntPtr GetWindow(IntPtr hWnd, uint cmd);
    [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr hWnd, int attr, out int value, int size);
    const uint GW_OWNER = 4;
    const int DWMWA_CLOAKED = 14;

    public record Entry(int pid, string process, string title);

    public static List<Entry> List()
    {
        var self = Environment.ProcessId;
        var result = new List<Entry>();
        EnumWindows((h, _) =>
        {
            if (!IsWindowVisible(h) || GetWindow(h, GW_OWNER) != IntPtr.Zero) return true;
            if (DwmGetWindowAttribute(h, DWMWA_CLOAKED, out var cloaked, 4) == 0 && cloaked != 0) return true;
            int len = GetWindowTextLength(h);
            if (len == 0) return true;
            var sb = new StringBuilder(len + 1);
            GetWindowText(h, sb, sb.Capacity);
            GetWindowThreadProcessId(h, out var pid);
            if (pid == self) return true;
            string name;
            try { name = Process.GetProcessById((int)pid).ProcessName + ".exe"; } catch { return true; }
            if (name is "explorer.exe" && sb.ToString() == "Program Manager") return true;
            result.Add(new Entry((int)pid, name, sb.ToString()));
            return true;
        }, IntPtr.Zero);
        return result;
    }
}

// Windows 10 2004+ process loopback: 지정 프로세스와 그 자식 프로세스들이 내는 소리만 캡처한다.
// 브라우저는 소리를 자식(오디오 서비스) 프로세스에서 내므로 프로세스 트리 포함 모드를 쓴다.
// AUTOCONVERTPCM 으로 OS 가 16 kHz mono PCM16 으로 바로 변환해 준다.
sealed class ProcessLoopbackCapture : IWaveIn
{
    const string VirtualDevice = @"VAD\Process_Loopback";
    static readonly Guid IID_IAudioClient = new("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2");
    const int AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000;
    const int AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = unchecked((int)0x80000000);
    const int AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000;

    readonly AudioClient _client;
    Thread? _thread;
    volatile bool _running;

    public WaveFormat WaveFormat { get; set; } = new WaveFormat(16000, 16, 1);
    public event EventHandler<WaveInEventArgs>? DataAvailable;
    public event EventHandler<StoppedEventArgs>? RecordingStopped;

    ProcessLoopbackCapture(AudioClient client) => _client = client;

    public static async Task<ProcessLoopbackCapture> CreateAsync(int pid)
    {
        var p = new ActivationParams { ActivationType = 1 /* PROCESS_LOOPBACK */, TargetProcessId = pid, ProcessLoopbackMode = 0 /* INCLUDE_TARGET_PROCESS_TREE */ };
        int size = Marshal.SizeOf<ActivationParams>();
        IntPtr blob = Marshal.AllocHGlobal(size);
        IntPtr pv = Marshal.AllocHGlobal(Marshal.SizeOf<PropVariantBlob>());
        try
        {
            Marshal.StructureToPtr(p, blob, false);
            Marshal.StructureToPtr(new PropVariantBlob { vt = 65 /* VT_BLOB */, cbSize = size, pBlobData = blob }, pv, false);
            var handler = new CompletionHandler();
            ActivateAudioInterfaceAsync(VirtualDevice, IID_IAudioClient, pv, handler, out _);
            var iface = await handler.Result.Task.WaitAsync(TimeSpan.FromSeconds(5));
            var client = new AudioClient((IAudioClient)iface);
            var capture = new ProcessLoopbackCapture(client);
            var flags = unchecked((AudioClientStreamFlags)(AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY));
            client.Initialize(AudioClientShareMode.Shared, flags, 2_000_000, 0, capture.WaveFormat, Guid.Empty);
            return capture;
        }
        finally
        {
            Marshal.FreeHGlobal(pv);
            Marshal.FreeHGlobal(blob);
        }
    }

    public void StartRecording()
    {
        _client.Start();
        _running = true;
        _thread = new Thread(Loop) { IsBackground = true };
        _thread.Start();
    }

    void Loop()
    {
        Exception? error = null;
        try
        {
            var cap = _client.AudioCaptureClient;
            int bpf = WaveFormat.BlockAlign;
            var buf = new byte[WaveFormat.AverageBytesPerSecond];
            while (_running)
            {
                int packet = cap.GetNextPacketSize();
                while (packet > 0)
                {
                    IntPtr data = cap.GetBuffer(out int frames, out var bufferFlags);
                    int bytes = frames * bpf;
                    if (buf.Length < bytes) buf = new byte[bytes];
                    if ((bufferFlags & AudioClientBufferFlags.Silent) != 0) Array.Clear(buf, 0, bytes);
                    else Marshal.Copy(data, buf, 0, bytes);
                    cap.ReleaseBuffer(frames);
                    DataAvailable?.Invoke(this, new WaveInEventArgs(buf, bytes));
                    packet = cap.GetNextPacketSize();
                }
                Thread.Sleep(10);
            }
        }
        catch (Exception e)
        {
            error = e;
        }
        RecordingStopped?.Invoke(this, new StoppedEventArgs(error));
    }

    public void StopRecording()
    {
        _running = false;
        _thread?.Join(1000);
        _client.Stop();
    }

    public void Dispose() => _client.Dispose();

    [StructLayout(LayoutKind.Sequential)]
    struct ActivationParams { public int ActivationType; public int TargetProcessId; public int ProcessLoopbackMode; }

    [StructLayout(LayoutKind.Sequential)]
    struct PropVariantBlob { public ushort vt; public ushort r1, r2, r3; public int cbSize; public IntPtr pBlobData; }

    [DllImport("Mmdevapi.dll", ExactSpelling = true, PreserveSig = false)]
    static extern void ActivateAudioInterfaceAsync(
        [MarshalAs(UnmanagedType.LPWStr)] string deviceInterfacePath,
        [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
        IntPtr activationParams,
        IActivateAudioInterfaceCompletionHandler completionHandler,
        out IActivateAudioInterfaceAsyncOperation activationOperation);

    [ComImport, Guid("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAgileObject { }

    [ComVisible(true)]
    sealed class CompletionHandler : IActivateAudioInterfaceCompletionHandler, IAgileObject
    {
        public readonly TaskCompletionSource<object> Result = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public void ActivateCompleted(IActivateAudioInterfaceAsyncOperation op)
        {
            op.GetActivateResult(out int hr, out object iface);
            if (hr < 0) Result.TrySetException(Marshal.GetExceptionForHR(hr)!);
            else Result.TrySetResult(iface);
        }
    }
}
