#!/bin/bash

set -euxo pipefail

iprof_dir=$(dirname $(which iprof))
lib_dir=${iprof_dir}/../lib

trace_no_analysis="$1"
outdir="$2"
bt2cmd="babeltrace2 --plugin-path=${lib_dir}"

record="perf record --call-graph=dwarf"

# trace to_interval
$record -o "$outdir/to_interval.data" $bt2cmd \
  --component=filter.zeinterval.interval \
  --component=sink.utils.dummy \
  "$trace_no_analysis"

# save interval data, trace to_agg
interval_ctf="$outdir/ctf-interval"
$bt2cmd --component=filter.zeinterval.interval \
  --output-format=ctf --output="$interval_ctf" \
  "$trace_no_analysis"

$record -o "$outdir/to_aggreg.data" $bt2cmd \
  --component=filter.btx_aggreg.aggreg \
  --component=sink.utils.dummy \
  "$interval_ctf/trace"

# save aggreg data, trace tally
aggreg_ctf="$outdir/ctf-aggreg"
$bt2cmd --component=filter.btx_aggreg.aggreg \
  --output-format=ctf --output="$aggreg_ctf" \
  "$interval_ctf/trace"

$record -o "$outdir/to_tally.data" $bt2cmd \
  --component=sink.btx_tally.tally \
  "$aggreg_ctf/trace"
