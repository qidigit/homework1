/* MPI ring communication
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "util.h"

int main( int argc, char *argv[])
{
    int rank, tag, size, orig, dest;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2 || argv[1] <= 0) {
        fprintf(stderr, "Input arguments not valid!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atol(argv[1]);
    int message_out = 0;
    int message_in = 0;
    tag = 99;


    // setup time counter
    timestamp_type time1, time2;
    get_timestamp(&time1);

    int i;
    for(i = 0; i < N; i++) {
        if(rank == 0) {
            dest = rank + 1;
            orig = size - 1;
            message_out = message_in + rank;

            MPI_Send(&message_out, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            MPI_Recv(&message_in, 1, MPI_INT, orig, tag, MPI_COMM_WORLD, &status);
        } else if(rank != size-1) {
            dest = rank + 1;
            orig = rank - 1;

            MPI_Recv(&message_in, 1, MPI_INT, orig, tag, MPI_COMM_WORLD, &status);
            message_out = message_in + rank;
            MPI_Send(&message_out, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        } else {
            dest = 0;
            orig = rank - 1;

            MPI_Recv(&message_in, 1, MPI_INT, orig, tag, MPI_COMM_WORLD, &status);
            message_out = message_in + rank;
            MPI_Send(&message_out, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }

//        printf("rank %d in loop %d received the message %d;\n", rank, i+1, message_out);
    }
    get_timestamp(&time2);
    double run_time = timestamp_diff_in_seconds(time1, time2);
    double comm_time = run_time / (double) N / (double) size;

    if(rank == 0) {
        printf("Total elapsed time is %f seconds.\n", run_time);
        printf("The estimated latency on this system is %.10f seconds.\n", comm_time);
        printf("The final received message on rank %d after %d loops is: %d.\n", rank, N, message_in);
    }


    MPI_Finalize();
    return 0;
}
