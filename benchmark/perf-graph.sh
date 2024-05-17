#!/bin/bash

set -euxo pipefail

data="$1"
folded="${data}.folded"

perf script -i "$data" | inferno-collapse-perf > "$folded"
cat "$folded" | inferno-flamegraph > "${data}.svg"
