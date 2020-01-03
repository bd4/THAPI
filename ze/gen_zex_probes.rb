require_relative 'ze_model'
require_relative 'gen_probe_base.rb'

provider = :lttng_ust_ze

puts <<EOF
#include <zex_api.h>
EOF

$zex_commands.each { |c|
  next if c.parameters && c.parameters.length > LTTNG_USABLE_PARAMS
  $tracepoint_lambda.call(provider, c, :start)
  $tracepoint_lambda.call(provider, c, :stop)
}


