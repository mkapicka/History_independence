https://www.metacentrum.cz/cs/Sluzby/

 Prerequisite (on your Mac — sweep.sh changed since your failed run)

  cd ~/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley
  rsync -av --exclude 'results' --exclude 'figures' --exclude '.DS_Store' hd/ meta:~/hd/

or

  cd ~/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley
  rsync -av --exclude 'results' --exclude 'figures' --exclude '.DS_Store' hdinf/ meta:~/hdinf/


  # 1. Log in

  ssh meta  (pwd rds@T*u4PZ%5fQ )

  # 2. Install the packages — on the frontend, once

  Compute nodes have no outbound network, so this cannot happen inside a job. This is what your JLD2 error was about.

  cd ~/hd
  module add julia/1.12.6
  export JULIA_DEPOT_PATH=$HOME/.julia
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.status()'
  julia --project=. -e 'using JLD2; println("JLD2 loads fine")'
  echo "HOME is: $HOME"

  Write down that HOME path — you'll need it only if step 3 shows the wrong depot.
    HOME is: /storage/praha1/home/mkapicka

  # 3. Trial run — 2 tasks

  qsub -J 1-2 sweep.sh
  qstat -u $USER                    # Q queued, R running, F finished, B begun

    pbs-m1.metacentrum.cz:
                                                                     Req'd  Req'd   Elap
    Job ID               Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
    -------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
    22815177[1].pbs-m1.* mkapicka q_4h     sweep_mu2  391322   1   8    8gb 04:00 X 00:01
    22815177[2].pbs-m1.* mkapicka q_4h     sweep_mu2  732387   1   8    8gb 04:00 R 00:13

  When both finish:
    qstat -xf '22815177[1]' | grep -i exit_status  # should be 0
    qstat -xf '22815177[2]' | grep -i exit_status  # should be 0

  cat sweep_mu2.o*.1                # stdout
  cat sweep_mu2.e*.1                # stderr, should be empty
  ls -la results/

  Four things to confirm: the depot: ... line points at the directory you instantiated in step 2; the summary says 1 converged; there's a saved to
  ... line; and results/ holds 2 .jld2 files.

  If the depot line is wrong, that's the only edit you'd need — set JULIA_DEPOT_OVERRIDE="/storage/.../home/YOURUSER/.julia" at the top of sweep.sh
  using the path from step 2, re-rsync, and repeat step 3.

  # 4. Full sweep — all tasks

  qsub sweep.sh
  qstat -u $USER

to check on time run:
  qstat -xt -u $USER

  # 5. Check every task converged

  grep -c "saved to" sweep_mu2.o*          # expect 10 files, 1 each
  grep -h "NOT CONVERGED\|FAILED\|WARNING" sweep_mu2.o* sweep_mu2.e*   # expect nothing
  ls results/*.jld2 | wc -l

  # 6. Bring the results back (from your Mac)

  rsync -av meta:~/hd/results/ \
    ~/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley/hd/results/

  or

  rsync -av meta:~/hdinf/results/ \
    ~/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley/hdinf/results/

  or

   rsync -av meta:~/hd_htm/results/ \
    ~/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley/hd_htm/results/


  # 7. Merge the ten one-point files

  using JLD2
  cd("/Users/mkapicka/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley/hd")
  files = sort(filter(f -> startswith(f, "sweep_mu2_n=1_") && endswith(f, ".jld2"),
                      readdir("results")))
  s   = [JLD2.load(joinpath("results", f))["sweep"] for f in files]
  mu2 = vcat((x.mu2 for x in s)...)
  W   = vcat((x.overallValueFunction for x in s)...)
  ok  = all(vcat((x.converged for x in s)...))

  Filenames embed mu2 to three decimals, so sorting by name gives ascending mu2.

  ---
  Two things to decide before step 4

  The script does not run your full-model resolution. MODEL_ARGS is currently

  mu1=0 alpha=0 tau=0.181 nS1=1 nS2=101 s_grid_method=:quantile

  so everything else falls back to HD_SETTINGS: nZ = 5, nEps = 5, labor_grid_size = 101. Your earlier single run used nZ=15 nEps=11
  labor_grid_size=151. If you want the sweep at that resolution, append those three to MODEL_ARGS — it's roughly 10× the work, about 20 minutes per
  point instead of 2, still comfortably inside the 4-hour walltime.

  Timing as configured is fast: nS1=1 (because mu1=0 collapses the s1 grid) makes each point cheaper than your 202-second smoke test despite nS2
  rising from 21 to 101. Expect a couple of minutes per task, so the whole sweep finishes in one scheduling round.


  # USEFUL COMMANDS
    qstat -u $USER                    # Q queued, R running, F finished, B begun
    qstat -xt -u $USER                 # check on time run on all jobs
    qdel '23314780[]'                 # kill all tasks in the array job. Take the job id from qstat -u $USER
    qselect -u $USER | xargs -r qdel  # kill all jobs in the queue for your user. Use with care.

Did the tasks actually succeed?
  for i in $(seq 1 20); do
    printf "%2d  " $i
    qstat -xf 23884398[$i]" 2>/dev/null | grep -i exit_status || echo "no record"
  done

  remove all logs
  rm sweep_mu2.[eo]*