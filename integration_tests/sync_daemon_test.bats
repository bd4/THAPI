#!/usr/bin/env bats

setup_file() {
   export MPIRUN=${MPIRUN:-mpirun}
   export MPICXX=${MPICXX:-mpicxx}
   export THAPI_SYNC_DAEMON=${THAPI_SYNC_DAEMON:-mpi}
   export TEST_EXE=${TEST_EXE:-clinfo}
   echo "# setup" >&3
}

@test "sync_daemon_${THAPI_SYNC_DAEMON}" {
   echo "# TEST_EXE=$TEST_EXE" >&3
   echo "# THAPI_SYNC_DAEMON=$THAPI_SYNC_DAEMON" >&3
   $MPIRUN -n 2 ./integration_tests/sync_daemon_test/test.sh
}
