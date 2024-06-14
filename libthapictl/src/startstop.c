#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lttng/lttng.h>

#include "thapi.h"
#include "thapi-ctl.h"
#include "backends.h"


#define THAPI_LTTNG_SESSION_ID_NAME "THAPI_LTTNG_SESSION_ID"
#define THAPI_CHANNEL_NAME "blocking-channel"

#define THAPI_CTL_DEFAULT_LOG_LEVEL THAPI_CTL_LOG_LEVEL_ERROR


static const char* thapi_ctl_level_str[5] = {
  "",
  "ERROR",
  "INFO",
  "WARN",
  "DEBUG"
};


static int thapi_ctl_log_level() {
  static int level = -1;
  if (level < 0) {
    char *log_level_str = getenv("THAPI_CTL_LOG_LEVEL");
    if (log_level_str == NULL || strlen(log_level_str) == 0) {
      level = THAPI_CTL_DEFAULT_LOG_LEVEL;
    } else {
      char *endptr;
      level = strtol(log_level_str, &endptr, 10);
      if (*endptr != '\0') {
        level = THAPI_CTL_DEFAULT_LOG_LEVEL;
      }
    }
  }
  return level;
}


static int* thapi_ctl_enabled_backends() {
  static int enabled_backends[] = {-1, -1, -1, -1, -1, -1, -1 };
  if (enabled_backends[BACKEND_UNKNOWN] == -1) {
    enabled_backends[BACKEND_UNKNOWN] = 0;
    enabled_backends[BACKEND_ZE] = (getenv("THAPI_LTTNG_BACKEND_ZE_ENABLED") != NULL);
    enabled_backends[BACKEND_OPENCL] =
      (getenv("THAPI_LTTNG_BACKEND_OPENCL_ENABLED") != NULL);
    enabled_backends[BACKEND_CUDA] = (getenv("THAPI_LTTNG_BACKEND_CUDA_ENABLED") != NULL);
    enabled_backends[BACKEND_OMP_TARGET_OPERATIONS] =
      (getenv("THAPI_LTTNG_BACKEND_OMP_TARGET_OPERATIONS_ENABLED") != NULL);
    enabled_backends[BACKEND_OMP] = (getenv("THAPI_LTTNG_BACKEND_OMP_ENABLED") != NULL);
    enabled_backends[BACKEND_HIP] = (getenv("THAPI_LTTNG_BACKEND_HIP_ENABLED") != NULL);
  }
  return enabled_backends;
}


static void log_events(struct lttng_handle* handle, const char *prefix) {
  int ret;
  struct lttng_event *event_list = NULL;
  struct lttng_event *event = NULL;
  int n_events = lttng_list_events(handle, THAPI_CHANNEL_NAME, &event_list);
  if (n_events < 0) {
    thapi_ctl_log(THAPI_CTL_LOG_LEVEL_ERROR, "[%s] error getting events: %d",
                  prefix, n_events);
  }
  thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "[%s] %d events", prefix, n_events);
  const char *filter_str;
  for (int i = 0; i < n_events; i++) {
    event = &event_list[i];
    thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "[%s] %s (enabled: %d)",
                  prefix, event->name, event->enabled);
    ret = lttng_event_get_filter_expression(event, &filter_str);
    if (ret) {
      thapi_ctl_log(THAPI_CTL_LOG_LEVEL_ERROR, "can't get filter expr %d", ret);
    } else if (filter_str) {
      thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "[%s]   + filter str '%s'",
                    prefix, filter_str);
      
    }
  }
  if (event_list != NULL)
    free(event_list);
}


void thapi_ctl_log(int log_level, const char *fmt, ...) {
  assert(log_level >= THAPI_CTL_LOG_LEVEL_ERROR && log_level <= THAPI_CTL_LOG_LEVEL_DEBUG);
  int config_level = thapi_ctl_log_level();
  if (log_level <= config_level) {
    fprintf(stderr, "libthapi-ctl[%s]: ", thapi_ctl_level_str[log_level]);
    va_list va;
    va_start(va, fmt);
    vfprintf(stderr, fmt, va);
    va_end(va);
    fprintf(stderr, "\n");
  }
}


static const char* thapi_ctl_lttng_session_id() {
  static char *session_id = NULL;
  if (session_id == NULL) {
    // first call, initialize static vars
    session_id = getenv(THAPI_LTTNG_SESSION_ID_NAME);
    if (session_id == NULL) {
      // not called with iprof
      session_id = "\0";
      thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG,
                    "No env var '%s', application not launched using iprof, disable",
                    THAPI_LTTNG_SESSION_ID_NAME);
    } else {
      thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG,
                    "THAPI session id: %s", session_id);
    }
  }
  return session_id;
}


static struct lttng_handle* thapi_ctl_create_lttng_handle(const char *session_id) {
  struct lttng_handle* handle = NULL;
  if (strlen(session_id) > 0) {
    struct lttng_domain dom;
    memset(&dom, 0, sizeof(dom));
    dom.type = LTTNG_DOMAIN_UST;
    dom.buf_type = LTTNG_BUFFER_PER_UID;
    handle = lttng_create_handle(session_id, &dom);
    if (handle == NULL) {
      thapi_ctl_log(THAPI_CTL_LOG_LEVEL_ERROR,
                   "Failed to create lttng handle for session '%s'",
                   session_id);

    }
  }
  return handle;
}


void thapi_ctl_destroy_lttng_handle(struct lttng_handle *handle) {
  if (handle != NULL) {
    lttng_destroy_handle(handle);
  }
}


int thapi_ctl_init() {
  const char *session_id = thapi_ctl_lttng_session_id();
  struct lttng_handle *handle = thapi_ctl_create_lttng_handle(session_id);
  if (handle == NULL)
    return 0;

  thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "init backends '%s'", session_id);
  // log_events(handle, "before");
  int *backends_enabled = thapi_ctl_enabled_backends();
  if (backends_enabled[BACKEND_CUDA]) {
    thapi_cuda_init(handle, THAPI_CHANNEL_NAME);
  }
  // log_events(handle, "after");
  thapi_ctl_destroy_lttng_handle(handle);
  return 0;
}


int thapi_ctl_start() {
  const char *session_id = thapi_ctl_lttng_session_id();
  struct lttng_handle *handle = thapi_ctl_create_lttng_handle(session_id);
  if (handle == NULL)
    return 0;

  thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "start tracing '%s'", session_id);
  // log_events(handle, "before");
  int *backends_enabled = thapi_ctl_enabled_backends();
  if (backends_enabled[BACKEND_CUDA]) {
    thapi_cuda_enable_tracing_events(handle, THAPI_CHANNEL_NAME);
  }
  log_events(handle, "after");
  thapi_ctl_destroy_lttng_handle(handle);
  return 0;
}


int thapi_ctl_stop() {
  const char *session_id = thapi_ctl_lttng_session_id();
  struct lttng_handle *handle = thapi_ctl_create_lttng_handle(session_id);
  if (handle == NULL)
    return 0;

  thapi_ctl_log(THAPI_CTL_LOG_LEVEL_DEBUG, "stop tracing '%s'",
                session_id);
  // log_events(handle, "before");

  int *backends_enabled = thapi_ctl_enabled_backends();
  if (backends_enabled[BACKEND_CUDA]) {
    thapi_cuda_disable_tracing_events(handle, THAPI_CHANNEL_NAME);
  }
  log_events(handle, "after");
  thapi_ctl_destroy_lttng_handle(handle);
  return 0;
}


void thapi_ctl_print_events() {
  const char *session_id = thapi_ctl_lttng_session_id();
  struct lttng_handle *handle = thapi_ctl_create_lttng_handle(session_id);
  if (handle == NULL)
    return;

  int ret;
  struct lttng_event *event_list = NULL;
  int n_events = lttng_list_events(handle, THAPI_CHANNEL_NAME, &event_list);
  if (n_events < 0) {
    thapi_ctl_log(THAPI_CTL_LOG_LEVEL_ERROR, "error getting events: %d",
                  n_events);
    return;
  }
  printf("%d events\n", n_events);
  const char *filter_str;
  for (int i = 0; i < n_events; i++) {
    printf("%s (enabled: %d)", event_list[i].name, event_list[i].enabled);
    ret = lttng_event_get_filter_expression(&event_list[i], &filter_str);
    if (ret != 0) {
      printf("Error getting filter expression: %d", ret);
    } else if (filter_str) {
      printf(" (filter '%s')", filter_str);
    }
    printf("\n");
  }
  if (event_list != NULL)
    free(event_list);
}
