#!/bin/bash

#!/bin/bash
trap 'killall' INT

killall() {
    trap '' INT TERM     # ignore INT and TERM while shutting down
    echo "**** Shutting down... ****"     # added double quotes
    kill -TERM 0         # fixed order, send TERM not INT
    wait
    echo DONE
}

# Launch 100 rq workers
for i in {1..40}
do
    rq worker SYNTH_DATA_GEN &
done

# Wait for all workers to finish
wait
