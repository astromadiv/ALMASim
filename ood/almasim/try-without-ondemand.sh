#!/bin/bash
# Rehearse the Open OnDemand app without a portal (Open OnDemand plan).
#
# Starts the console as a Slurm job the way the portal would: this app's own
# template/before.sh.erb and template/script.sh.erb, with stand-ins for
# OnDemand's find_port and create_passwd. Then it prints an SSH tunnel, which
# plays the portal's proxy, and the link the portal's Connect button opens.
#
#   bash ood/almasim/try-without-ondemand.sh                 # 1 CPU, 4 GB, 8 h
#   bash ood/almasim/try-without-ondemand.sh -p PART -A ACC  # sbatch options
#   bash ood/almasim/try-without-ondemand.sh --replace       # stop it, start anew
#   ALMASIM_SETUP=~/dev-env.sh bash ood/almasim/try-without-ondemand.sh
#
# One console per user: when one is already running (or starting), its tunnel
# and link are printed again instead of starting a second. Every console holds
# a Slurm job slot for 8 h, and three forgotten ones once kept an imaging
# worker queued behind the per-user job limit (2026-10-28). --replace stops
# it and starts a new one, e.g. after updating the install.
# ALMASIM_SETUP is the script that puts almasim on the PATH (default: the
# site's, as form.yml sets it). Each session keeps its files in
# ~/ondemand-try/<session>/, as OnDemand keeps them under ~/ondemand/data/.
# Stop a session with: scancel <job id>.
set -euo pipefail

replace=false
if [ "${1:-}" = "--replace" ]; then replace=true; shift; fi

app=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
form_setup=$(awk '/^  almasim_setup:/ {f = 1} f && /value:/ {gsub(/.*value: *"|".*/, ""); print; exit}' "$app/form.yml")
setup=${ALMASIM_SETUP:-$form_setup}
[ -r "$setup" ] || { echo "Cannot read the setup script $setup (set ALMASIM_SETUP)." >&2; exit 1; }
setup=$(cd "$(dirname "$setup")" && pwd)/$(basename "$setup")

# This user's consoles, newest first: "<job id> <session directory>".
running=$(squeue -h -u "$USER" -n almasim-try -t PENDING,RUNNING -o '%i %Z' | sort -rn)
if [ -n "$running" ] && $replace; then
    scancel $(cut -d' ' -f1 <<<"$running")
    echo "Stopped the running console(s): $(cut -d' ' -f1 <<<"$running" | xargs)"
    running=
fi
if [ -n "$running" ]; then
    read -r job session <<<"$(head -1 <<<"$running")"
    others=$(tail -n +2 <<<"$running" | cut -d' ' -f1 | xargs)
    echo "A console is already running: Slurm job $job, session $session."
    [ -z "$others" ] || echo "More consoles are running; stop them with: scancel $others"
fi

if [ -z "$running" ]; then
mkdir -p -m 700 ~/ondemand-try
session=$(mktemp -d ~/ondemand-try/session-XXXXXX)

# The job: OnDemand's helpers, then the app's own scripts.
cat > "$session/job.sh" <<JOB
#!/bin/bash
set -euo pipefail
find_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'; }
create_passwd() { python3 -c 'import secrets, string, sys; print("".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(int(sys.argv[1]))))' "\$1"; }
# The node's short name, as OnDemand's set_host is usually configured: the
# login node (here) and a portal host resolve compute nodes by it, not by
# the long name hostname gives.
host=\$(hostname -s)
source "$app/template/before.sh.erb"
source "$setup"
if ! almasim serve --managed --help >/dev/null 2>&1; then
    echo "\$(command -v almasim) predates the OnDemand app (no serve --managed):" \\
         "update that install, or set ALMASIM_SETUP to a newer one." > failed.txt
    exit 1
fi
( umask 077; printf 'host=%s\nport=%s\npassword=%s\n' "\$host" "\$port" "\$password" > connection.env )
sed 's|<%= context.almasim_setup %>|$setup|' "$app/template/script.sh.erb" > script.sh
exec bash script.sh
JOB

job=$(sbatch --parsable -J almasim-try -c 1 --mem=4G -t 08:00:00 \
    --chdir "$session" -o "$session/output.log" "$@" "$session/job.sh")
echo "Session $session: Slurm job $job; waiting for it to start..."
fi
while [ ! -f "$session/connection.env" ]; do
    if [ -f "$session/failed.txt" ] || ! squeue -h -j "$job" 2>/dev/null | grep -q .; then
        # Its last words, once the shared filesystem shows them here.
        for _ in $(seq 10); do
            ls "$session" >/dev/null
            [ -s "$session/failed.txt" ] || [ -s "$session/output.log" ] && break
            sleep 2
        done
        cat "$session/failed.txt" "$session/output.log" >&2 2>/dev/null || true
        echo "The session ended before the console started (job $job)." >&2
        exit 1
    fi
    sleep 3
done
source "$session/connection.env"

cat <<MSG

1. On your laptop, open the tunnel (it plays the portal's proxy):

   ssh -N -L 8765:${host}:${port} ${USER}@$(hostname -f)

2. In your browser, open (as the portal's Connect button would):

   http://localhost:8765/node/${host}/${port}/?token=${password}

The console runs as Slurm job ${job}; its log is ${session}/output.log.
Stop it with:  scancel ${job}
Start a new one instead (e.g. after updating the install) with --replace.
MSG
