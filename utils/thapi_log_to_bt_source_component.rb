#!/usr/bin/env ruby
#
# Take output of `babeltrace_thapi trace -c` and generate a C metababel
# callbacks SOURCE type plugin that will generate the same event stream.
#
# For use in creating test cases for unit testing, with input trace files that
# are easy to maintain.
#
# Does not yet handle complex struct args, so the output may not be identical when
# roundtripping through babeltrace_thapi.

require 'optparse'
require 'yaml'
require 'erb'
require 'time'
require 'set'

$empty_array_name = "EMPTY_ARRAY"

SOURCE_TEMPLATE = <<~TEXT.freeze
  /* Code generated by #{__FILE__} */

  #include <babeltrace2/babeltrace.h>
  #include <metababel/metababel.h>

  static const void** #{$empty_array_name};

  void btx_push_usr_messages(void *btx_handle, void *usr_data, btx_source_status_t *status) {
      <%- if not data.empty? and not data.first[:hostname].nil? -%>
      btx_downstream_set_environment_hostname(btx_handle, "<%= data.first[:hostname] %>");
      <%- end -%>

      <%- data.each do | entry | -%>
      btx_push_message_<%= entry[:name] %>(btx_handle<%= ', ' if not entry[:field_values].empty? %><%= 
       entry[:field_casts].zip(entry[:field_values].map{ |v| v.to_s }).map{ |cv| '(' + cv[0] + ')' + cv[1] }.join(', ')
      %>);
      <%- end -%>

      *status = BTX_SOURCE_END;
  }

  void btx_register_usr_callbacks(void *btx_handle) {
    btx_register_callbacks_push_usr_messages(btx_handle, &btx_push_usr_messages);
  }
TEXT

BASE_DATE = Time.utc(2024)

def get_values(name, value)
  if not name.end_with?("_vals") or not value.strip.start_with?("[")
    # simple value, do not modify
    return [value]
  end

  # value will look like '[ ]' or '[ value ]' or '[ val1, val2, ... ]'
  count = 0
  value = value.strip[1...-1].strip
  if value.empty?
    count = 0
    # use static const empty array defined in callback template
    value = $empty_array_name;
  else
    values = value.split(/, */)
    count = 1
    # TODO: handle multiple values
    value = values[0]
  end
  [count, value]
end

# expects commas separated list of name: value pairs
def parse_field_values(fields_string, exclude_fields)
  fields_string.split(/, */).filter_map do |nv_pair|
    name, value = nv_pair.split(': ', 2).map(&:strip)
    get_values(name, value) unless exclude_fields.member?(name)
  end.flatten
end

def get_fields(model, event_name, stream_class=0)
  fields = {}
  stream_classes = model[:stream_classes]
  sc = stream_classes[0]
  sc[:event_common_context_field_class][:members].each do |field_info|
    fields[field_info[:name]] = field_info[:field_class][:cast_type]
  end
  eclass = sc[:event_classes].find { |eclass| eclass[:name] == event_name }
  field_class = eclass[:payload_field_class][:members].to_h { |field_info|
     [ field_info[:name], field_info[:field_class][:cast_type] ]
  }
  fields.update(field_class)
end

def combine_date_time(date, time)
  Time.utc(date.year, date.month, date.mday,
           time.hour, time.min, time.sec, time.subsec)
end

# Example input line from babeltrace_thapi:
# 12:59:07.177888481 - gene-lxc - vpid: 415, vtid: 415 - lttng_ust_cuda:cuDevicePrimaryCtxRetain_exit: { cuResult: CUDA_SUCCESS, pctx_val: 0x000055c6244d84f0 }
def parse_event(model, line, exclude_fields)
  h = { hostname: nil,
        name: nil,
        field_values: [],
        field_casts: [] }

  parts = line.strip.split(' - ', 4)

  if parts.length == 4
    # backend model
    ts_string, h[:hostname], context, event = parts
    t = combine_date_time(BASE_DATE, Time.parse(ts_string))
    # Need to convert in nasosecond, store as first field
    h[:field_values] << ((t.to_i * 1_000_000_000) + t.nsec)
    h[:field_casts] << 'int64_t'
  elsif parts.length == 2
    # interval model, no hostname or ts
    context, event = parts
  else
    raise "unable to parse input line, length #{parts.length}: #{line.strip}"
  end

  # context fields are next after timestamp
  h[:field_values] += parse_field_values(context, exclude_fields)

  # handle event name and fields
  event_name, event_fields = event.split('{', 2)
  # strip of trailing space and colon, and replace non alpha-numeric
  # with underscore
  event_name = event_name.strip.chop
  h[:name] = event_name.gsub(/[^0-9A-Za-z-]/, '_')

  # finally add the event fields
  # remove trailing } and space from fields
  if not event_fields.nil?
    event_fields = event_fields.chop.strip
    h[:field_values] += parse_field_values(event_fields, exclude_fields)
  end
  h[:field_casts] += get_fields(model, event_name).values
  if (h[:field_values].length != h[:field_casts].length)
    puts "WARN: field values / casts mismatch"
    pp h[:field_values]
    pp h[:field_casts]
  end
  h
end

def parse_log(model, input_path, exclude_fields)
  File.open(input_path, 'r') do |file|
    file.each_line.map do |line|
      parse_event(model, line, exclude_fields)
    end
  end
end

def render_and_save(data, output_path)
  renderer = ERB.new(SOURCE_TEMPLATE, trim_mode: '-')
  output = renderer.result(binding)
  File.write(output_path, output, mode: 'w')
end

DOCS = <<-DOCS.freeze
  Usage: #{$0}.rb [options]

  Example:
    ruby #{$0} -y stream_classes.yaml -i btx_log.txt -o callbacks.c
DOCS

# Display help if no arguments.
ARGV << '-h' if ARGV.empty?

options = {}

OptionParser.new do |opts|
  opts.banner = DOCS

  opts.on('-h', '--help', 'Prints this help') do
    puts opts
    exit
  end

  opts.on('-y', '--model PATH', '[Mandatory] Path to btx_model.yaml') do |p|
    options[:model_path] = p
  end

  opts.on('-i', '--log PATH', '[Mandatory] Path to btx_log.txt.') do |p|
    options[:input_path] = p
  end

  opts.on('-o', '--output PATH', '[Mandatory] Path to the bt2 SOURCE file.') do |p|
    options[:output_path] = p
  end

  opts.on('-eFIELD_NAMES', '--exclude-fields=FIELD_NAMES',
          '[Optional] Exclude fields in callbacks (comma separated list)') do |e|
    options[:exclude_fields] = e.split(',').to_set
  end
end.parse!

raise OptionParser::MissingArgument if options[:output_path].nil?

model = YAML.load_file(options[:model_path])

data = options.key?(:input_path) ? parse_log(model, options[:input_path], options.fetch(:exclude_fields, Set[])) : []
render_and_save(data, options[:output_path])
