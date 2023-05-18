#load: main.cu
# 	module load compiler/cuda
# 	module load suite/nvidia-hpc-sdk
#	module load compiler/intel/2019u5/intelpython3
# 	qsub -I -P col380.cs1190335 -l select=1:ncpus=1:ngpus=1:centos=haswell -l walltime=00:20:00
# 	cd home/cse/btech/cs1190335/A4
#	cd/home/cse/btech/csxxxxxxx/
# 	-arch=sm_56

compile: main.cu
	nvcc -O3 -std=c++11 -arch=sm_35 main.cu -o exec

run: main.cu
	./exec inputFile1.bin inputFile2.bin outFile.bin

clean:
	rm -rf *.o exec