param(
    [Parameter(Mandatory = $true)] [string]$CommandJsonBase64,
    [Parameter(Mandatory = $true)] [string]$WorkingDirectory
)

$ErrorActionPreference = "Stop"
$source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class InsertAny3DJobRunner
{
    private const uint CREATE_SUSPENDED = 0x00000004;
    private const uint CREATE_UNICODE_ENVIRONMENT = 0x00000400;
    private const uint INFINITE = 0xFFFFFFFF;
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    private const int JobObjectExtendedLimitInformation = 9;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct STARTUPINFO
    {
        public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
        public int dwX; public int dwY; public int dwXSize; public int dwYSize;
        public int dwXCountChars; public int dwYCountChars; public int dwFillAttribute; public int dwFlags;
        public short wShowWindow; public short cbReserved2; public IntPtr lpReserved2;
        public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION
    {
        public IntPtr hProcess; public IntPtr hThread; public int dwProcessId; public int dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass; public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount; public ulong WriteOperationCount; public ulong OtherOperationCount;
        public ulong ReadTransferCount; public ulong WriteTransferCount; public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int kind,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION information, uint length);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string applicationName, StringBuilder commandLine,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles, uint creationFlags,
        IntPtr environment, string currentDirectory, ref STARTUPINFO startupInfo,
        out PROCESS_INFORMATION processInformation);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);

    public static int Run(string[] command, string workingDirectory)
    {
        if (command == null || command.Length == 0 || String.IsNullOrWhiteSpace(command[0]))
            throw new ArgumentException("Command must contain an executable.");
        IntPtr job = IntPtr.Zero;
        PROCESS_INFORMATION process = new PROCESS_INFORMATION();
        bool processCreated = false;
        try
        {
            job = CreateJobObject(IntPtr.Zero, null);
            Check(job != IntPtr.Zero, "CreateJobObject");
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            Check(SetInformationJobObject(job, JobObjectExtendedLimitInformation, ref limits,
                (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION))), "SetInformationJobObject");

            STARTUPINFO startup = new STARTUPINFO();
            startup.cb = Marshal.SizeOf(typeof(STARTUPINFO));
            StringBuilder commandLine = new StringBuilder(BuildCommandLine(command));
            Check(CreateProcess(null, commandLine, IntPtr.Zero, IntPtr.Zero, false,
                CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT, IntPtr.Zero, workingDirectory,
                ref startup, out process), "CreateProcess");
            processCreated = true;
            Check(AssignProcessToJobObject(job, process.hProcess), "AssignProcessToJobObject");
            if (ResumeThread(process.hThread) == UInt32.MaxValue) ThrowLastError("ResumeThread");
            WaitForSingleObject(process.hProcess, INFINITE);
            uint exitCode;
            Check(GetExitCodeProcess(process.hProcess, out exitCode), "GetExitCodeProcess");
            return unchecked((int)exitCode);
        }
        catch
        {
            if (processCreated && process.hProcess != IntPtr.Zero) TerminateProcess(process.hProcess, 1);
            throw;
        }
        finally
        {
            if (process.hThread != IntPtr.Zero) CloseHandle(process.hThread);
            if (process.hProcess != IntPtr.Zero) CloseHandle(process.hProcess);
            if (job != IntPtr.Zero) CloseHandle(job);
        }
    }

    private static string BuildCommandLine(string[] command)
    {
        StringBuilder result = new StringBuilder();
        foreach (string item in command)
        {
            if (result.Length > 0) result.Append(' ');
            result.Append(Quote(item ?? String.Empty));
        }
        return result.ToString();
    }

    private static string Quote(string value)
    {
        if (value.Length > 0 && value.IndexOfAny(new char[] { ' ', '\t', '\n', '\v', '"' }) < 0) return value;
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char current in value)
        {
            if (current == '\\') { slashes++; continue; }
            if (current == '"')
            {
                result.Append('\\', slashes * 2 + 1); result.Append('"'); slashes = 0; continue;
            }
            result.Append('\\', slashes); slashes = 0; result.Append(current);
        }
        result.Append('\\', slashes * 2); result.Append('"');
        return result.ToString();
    }

    private static void Check(bool success, string operation) { if (!success) ThrowLastError(operation); }
    private static void ThrowLastError(string operation)
    {
        int error = Marshal.GetLastWin32Error();
        throw new Win32Exception(error, operation + " failed (Win32 " + error + ")");
    }
}
'@

try {
    $CommandJson = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($CommandJsonBase64))
    $command = [string[]]($CommandJson | ConvertFrom-Json)
    if ($command.Count -eq 0) { throw "CommandJson must contain at least one argument" }
    Add-Type -TypeDefinition $source -Language CSharp
    $exitCode = [InsertAny3DJobRunner]::Run($command, $WorkingDirectory)
    exit $exitCode
}
catch {
    [Console]::Error.WriteLine("WINDOWS_JOB_RUNNER_ERROR: " + $_.Exception.Message)
    exit 125
}
