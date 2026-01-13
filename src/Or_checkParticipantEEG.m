function report = Or_checkParticipantEEG(participantPrefix, Fs, varargin)
% test change by Orwa
% test save 
% Or_checkParticipantEEG  Quality checks for all EEG variables of one participant.
%
% USAGE:
%   report = Or_checkParticipantEEG("Design_1", 250);
%   report = Or_checkParticipantEEG("Creativity_12", 500, 'DoPlots', false);
%
% INPUTS:
%   participantPrefix : "Design_1" or "Creativity_5" etc.
%   Fs                : sampling rate (Design dataset often 250 Hz, Creativity often 500 Hz)
%
% OPTIONAL name-value pairs:
%   'WinSec'       : window length in seconds (default 2)
%   'ThAbs'        : max(|x|) threshold in uV (default 100)
%   'ThPTP'        : peak-to-peak threshold in uV (default 200)
%   'BadChFrac'    : fraction of channels allowed to be bad in a window (default 0.25)
%   'DoPlots'      : true/false (default true)
%   'MaxVarsPlot'  : maximum number of variables to plot PSD/time for (default 6)
%
% OUTPUT:
%   report : struct containing per-variable results + summary

% ---------------------- parse inputs ----------------------
p = inputParser;
p.addRequired('participantPrefix', @(x) isstring(x) || ischar(x));
p.addRequired('Fs', @(x) isnumeric(x) && isscalar(x) && x > 0);

p.addParameter('WinSec', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('ThAbs', 100, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('ThPTP', 200, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('BadChFrac', 0.25, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
p.addParameter('DoPlots', true, @(x) islogical(x) && isscalar(x));
p.addParameter('MaxVarsPlot', 6, @(x) isnumeric(x) && isscalar(x) && x >= 0);

p.parse(participantPrefix, Fs, varargin{:});
opt = p.Results;

participantPrefix = string(participantPrefix);

% ---------------------- find variables ----------------------
vars = evalin('base','whos');
names = string({vars.name});

mask = startsWith(names, participantPrefix + "_");
varNames = names(mask);

if isempty(varNames)
    error('No variables found in base workspace starting with: %s_', participantPrefix);
end

% Keep only 2D matrices with 63 rows (typical EEG)
good = false(size(varNames));
for i = 1:numel(varNames)
    sz = evalin('base', sprintf('size(%s)', varNames(i)));
    good(i) = numel(sz)==2 && sz(1)==63;
end
varNames = varNames(good);

if isempty(varNames)
    error('Found variables for %s but none matched size [63 x T].', participantPrefix);
end

% ---------------------- analyze each variable ----------------------
report = struct();
report.participantPrefix = participantPrefix;
report.Fs = Fs;
report.options = opt;
report.variables = struct();

fprintf('\n=== Or_checkParticipantEEG: %s | Fs=%g Hz ===\n', participantPrefix, Fs);
fprintf('Found %d EEG variables (63 x T).\n', numel(varNames));

% For overall summary
all_badWindows = 0;
all_totalWindows = 0;
flag_abs = 0;
flag_ptp = 0;

% plotting counters
plotsDone = 0;

for k = 1:numel(varNames)
    vname = varNames(k);
    EEG = evalin('base', vname); % [63 x T]
    EEG = double(EEG);

    [nCh, nS] = size(EEG);
    durationSec = nS / Fs;

    % ---- channel amplitude checks ----
    ptp = max(EEG,[],2) - min(EEG,[],2);
    mx  = max(abs(EEG),[],2);

    badCh_abs = find(mx > opt.ThAbs);
    badCh_ptp = find(ptp > opt.ThPTP);

    if ~isempty(badCh_abs), flag_abs = flag_abs + 1; end
    if ~isempty(badCh_ptp), flag_ptp = flag_ptp + 1; end

    % ---- window-level badness ----
    winSamp = round(opt.WinSec * Fs);
    nWin = floor(nS / winSamp);
    badWinMask = false(1, nWin);
    badChCount = zeros(1, nWin);

    for w = 1:nWin
        idx = (w-1)*winSamp + (1:winSamp);
        Xw = EEG(:, idx);

        ptp_w = max(Xw,[],2) - min(Xw,[],2);
        mx_w  = max(abs(Xw),[],2);

        badCh_w = (ptp_w > opt.ThPTP) | (mx_w > opt.ThAbs);
        badChCount(w) = sum(badCh_w);

        if (badChCount(w)/nCh) > opt.BadChFrac
            badWinMask(w) = true;
        end
    end

    badWins = find(badWinMask);
    fracBadWins = numel(badWins) / max(nWin,1);

    all_badWindows = all_badWindows + numel(badWins);
    all_totalWindows = all_totalWindows + nWin;

    % ---- PSD quick check on one channel (ch=1) ----
    ch = 1;
    x = EEG(ch,:);
    x = x - mean(x);
    % Use Welch (works without extra toolboxes)
    % --- FFT-based PSD (no toolboxes) ---
    N = numel(x);
    x = x(:) - mean(x);
    
    X = fft(x);
    Pxx = (abs(X).^2) / (Fs*N);
    f = (0:N-1)*(Fs/N);
    
    % keep only positive frequencies
    half = floor(N/2);
    pxx = Pxx(1:half);
    f = f(1:half);

    % Simple line-noise indicator at 50 Hz (Israel) if within range
    lineFreq = 50;
    [~, idx50] = min(abs(f - lineFreq));
    linePower = pxx(idx50);

    % ---- store in report ----
    R = struct();
    R.size = [nCh, nS];
    R.durationSec = durationSec;
    R.maxAbsPerCh = mx;
    R.ptpPerCh = ptp;
    R.badChannels_abs = badCh_abs(:)';
    R.badChannels_ptp = badCh_ptp(:)';
    R.nWindows = nWin;
    R.badWindows = badWins(:)';
    R.fracBadWindows = fracBadWins;
    R.linePower50Hz_ch1 = linePower;

    report.variables.(matlab.lang.makeValidName(vname)) = R;

    % ---- print compact line ----
    fprintf('%-18s | T=%6.1fs | badCh(abs)=%2d | badCh(ptp)=%2d | badWin=%3d/%3d (%.1f%%)\n', ...
        vname, durationSec, numel(badCh_abs), numel(badCh_ptp), numel(badWins), nWin, 100*fracBadWins);

    % ---- optional plots (limit how many) ----
    if opt.DoPlots && plotsDone < opt.MaxVarsPlot
        plotsDone = plotsDone + 1;

        % time series: 6 channels, first 10 sec
        secToPlot = min(10, durationSec);
        N = min(nS, round(secToPlot*Fs));
        t = (0:N-1)/Fs;
        chList = round(linspace(1,nCh,6));

        figure('Name', char(vname) + " - Time series");
        for i = 1:numel(chList)
            subplot(numel(chList),1,i);
            plot(t, EEG(chList(i),1:N));
            ylabel(sprintf('Ch %d', chList(i)));
            if i==1, title(sprintf('%s | first %.1f sec', vname, secToPlot)); end
            if i==numel(chList), xlabel('Time (s)'); end
        end

        % PSD: channel 1
        figure('Name', char(vname) + " - PSD");
        semilogy(f, pxx);
        xlim([0 60]);
        grid on;
        xlabel('Hz'); ylabel('PSD');
        title(sprintf('%s | PSD (ch1) | Fs=%g', vname, Fs));

        % bad channels per window
        if nWin > 0
            figure('Name', char(vname) + " - Window badness");
            bar(badChCount);
            hold on;
            yline(opt.BadChFrac*nCh,'r--','Threshold');
            xlabel('Window #'); ylabel('# bad channels');
            title(sprintf('%s | bad channels per %.1fs window', vname, opt.WinSec));
        end
    end
end

% ---------------------- summary ----------------------
report.summary = struct();
report.summary.nVariables = numel(varNames);
report.summary.flag_absVars = flag_abs;
report.summary.flag_ptpVars = flag_ptp;
report.summary.totalWindows = all_totalWindows;
report.summary.badWindows = all_badWindows;
report.summary.badWindowsPercent = 100 * (all_badWindows / max(all_totalWindows,1));

fprintf('\n--- Summary for %s ---\n', participantPrefix);
fprintf('Variables analyzed: %d\n', report.summary.nVariables);
fprintf('Vars with any badCh(abs): %d | Vars with any badCh(ptp): %d\n', flag_abs, flag_ptp);
fprintf('Bad windows overall: %d / %d (%.2f%%)\n\n', ...
    all_badWindows, all_totalWindows, report.summary.badWindowsPercent);
end
