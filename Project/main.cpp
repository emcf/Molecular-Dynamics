// include all of the required header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include "vector3d.h"
#include "savebmp.h"
#include "properties.h"

// function to convert an array of doubles into an array of vec3 objects
vec3* convertArray (double* data, int size){
	vec3 *output = (vec3 *)malloc ((size/3) * sizeof(vec3));
	for (int i = 0; i < size; i+=3){
		output[i/3] = vec3(data[i], data[i+1], data[i+2]);
	}
	return output;
}

// function to calculate force for particles from index first to last
void calculateForce(vec3 *position, vec3 *force,  double *mass, int total, int first, int last){
	int i, j;
	double factor;
#pragma omp for
	for (i = first; i < last; i++){
		force[i - first] = vec3();
		for (j = 0; j < total; j++){
			if (j != i){
				// calculate distance between particle i and j and cube it
				factor = (position[i] - position[j]).Magnitude();
				factor = pow(factor,3);
				// divide mass of particle j by the factor
				factor = mass[j]/factor;
				// multiply the factor by the difference in the positions of the particles and add it to the force
				force[i-first] += (position[i] - position[j]) * factor; 
			}
		}	
		// multiply by -Gm and mass of particle i
		force[i-first] = force[i-first]*(mass[i]* -0.00000000006673); 
	}
	
}

// calculate the new positions and velocity required for a step
void calculateStep(vec3 *position, vec3 *velocity, vec3 *force, double *mass, double timeSteps, int start, int total, int height, int width, int depth){
#pragma omp for
	for (int i = 0; i< total; ++i){
		// add the calculated force times (stepsize/mass of particle i) to the new velocity of particle i
		velocity[i] += force[i] * (timeSteps / mass[i+start]) ;
		// add the velocity times the stepsize to get the new position of particle i
		position[i] += velocity[i] * timeSteps;
	}
}

int main(int argc, char* argv[]){
	// check if the right number of command line arguments are used
	if( argc != 10){
		printf("Usage: %s numParticlesLight numParticleMedium numParticleHeavy numSteps subSteps timeSubStep imageWidth imageHeight imageFilenamePrefix\n", argv[0]);
		return 0;
	}

	// initialize the mpi stuff
	MPI_Init(&argc,&argv);
	int p, my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	//initialize the variables
	double timeSteps;
	int light, medium, heavy, steps, subSteps, width, height, depth;
	light = atoi(argv[1]);
	medium = atoi(argv[2]);
	heavy = atoi(argv[3]);
	steps = atoi(argv[4]);
	subSteps = atoi(argv[5]);
	timeSteps = atof(argv[6]);
	width = atoi(argv[7]);
	height = atoi(argv[8]);
	depth = atoi(argv[7]);

	// calculate the offsets and distributions to be used in the Scatterv later
	int total = light + medium + heavy;
	int *offset = (int *) malloc (p * sizeof(int));
	int *distribution = (int *) malloc(p * sizeof(int));
	int portion;
	int displacment = 0;
	for (int i = 0; i< p; i++){
		if (i <  total % p){
			distribution[i] = ((total / p) + 1)*3;
		}
		else{
			distribution[i] = (total / p) * 3;
		}
		offset[i] = displacment;
		displacment += distribution[i];
		if (my_rank == i)
			// portion is also used later in the MPI communication
			portion = distribution[i] / 3;
	}

	// initialize variables used for the actual calculations
	unsigned char* image = (unsigned char *) malloc (sizeof (unsigned char) * 3 * (width*height));
	vec3 *force = (vec3 *)malloc ((portion) * sizeof(vec3));
	vec3 *position = (vec3 *)malloc ((total) * sizeof(vec3));
	vec3 *velocity;
	double *mass = (double *)malloc ((total) * sizeof(vec3));
	double *runtimes = (double *) malloc ((steps) * sizeof(double));

	//root node randomly generates all of the particles values
	if(my_rank == 0){
		srand48(time(NULL));
		velocity = (vec3 *)malloc ((total) * sizeof(vec3));
		for(int i = 0; i<light; i ++){
			double vx = drand48() * (velocityLightMax - velocityLightMin + 1) + velocityLightMin;
			double vy = drand48() * (velocityLightMax - velocityLightMin + 1) + velocityLightMin;
			double vz = drand48() * (velocityLightMax - velocityLightMin + 1) + velocityLightMin;
			// these if statements are to give an even distribution of initial particle velocities so that the video looks better
			if (i%2 == 0){
				vx = -vx;
			}
			if (i%3 == 0){
				vy = -vy;
			}
			if (i%4 ==0){
				vz = -vz;
			}
			double sx = drand48() * (width);
			double sy = drand48() * (height);
			double sz = drand48() * (depth);
			double massp = drand48() * (massLightMax-massLightMin+1)+massLightMin;

			// assign the values to corresponding arrays
			position[i] = vec3 (sx, sy, sz);
			velocity[i] = vec3 (vx, vy, vz);
			mass[i] = massp;
		}
		for(int i = light; i<light + medium; i ++){
			double vx = drand48() * (velocityMediumMax - velocityMediumMin + 1)+velocityMediumMin;
			double vy = drand48() * (velocityMediumMax - velocityMediumMin + 1)+velocityMediumMin;
			double vz = drand48() * (velocityMediumMax - velocityMediumMin + 1)+velocityMediumMin;
			// these if statements are to give an even distribution of initial particle velocities so that the video looks better
	                if (i%2 == 0){     
                                vx = -vx;  
                        }
                        if (i%3 == 0){
                                vy = -vy;
                        }
                        if (i%4 ==0){
                                vz = -vz;
                        }   
			double sx = drand48() * (width);
			double sy = drand48() * (height);
			double sz = drand48() * (depth);
			double massp = drand48() * (massMediumMax-massMediumMin+1)+massMediumMin;

			// assign the values to corresponding arrays
			position[i] = vec3 (sx, sy, sz);
			velocity[i] = vec3 (vx, vy, vz);
			mass[i] = massp;
		}
		for(int i = light + medium; i<total; i++){
			double vx = drand48() * (velocityHeavyMax - velocityHeavyMin + 1)+velocityHeavyMin;
			double vy = drand48() * (velocityHeavyMax - velocityHeavyMin + 1)+velocityHeavyMin;
			double vz = drand48() * (velocityHeavyMax - velocityHeavyMin + 1)+velocityHeavyMin;
			// these if statements are to give an even distribution of initial particle velocities so that the video looks better
                        if (i%2 == 0){     
                                vx = -vx;  
                        }
                        if (i%3 == 0){
                                vy = -vy;
                        }
                        if (i%4 ==0){
                                vz = -vz;
                        }   
			double sx = drand48() * (width);
			double sy = drand48() * (height);
			double sz = drand48() * (depth);
			double massp = drand48() * (massHeavyMax- massHeavyMin+1)+massHeavyMin;

			// assign values to corresponding arrays
			position[i] = vec3 (sx, sy, sz);
			velocity[i] = vec3 (vx, vy, vz);
			mass[i] = massp;
		}
	}

	// translate the array of vec3 objects into arrays of doubles and create buffers
	double * positionD = &position[0].x;
	double * velocityD = &velocity[0].x;
	double * my_velocity = (double *)malloc ((portion) * 3 * sizeof(double));

	// broadcast all of the positions and masses since every position and mass is needed to calculate a single particles new values
	MPI_Bcast(positionD, (total)*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(mass, total, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// scatter the velocities so that each node only receives the velocities it needs for it's calculations
	MPI_Scatterv(velocityD, distribution, offset, MPI_DOUBLE, my_velocity, (portion) * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// turn arrays of doubles received into arrays of vec3 objects for the function calls later on
	vec3 * my_position = (vec3 *)malloc ((int)(portion) * sizeof (vec3));
	if(my_rank != 0)
		velocity = convertArray(my_velocity, (portion) * 3);
	my_position = convertArray((positionD + (offset[my_rank])), distribution[my_rank] );

	// variables for timing and file naming
	double start, end;
	int count = 1;

	// start calculating substeps
	for(int i = 1; i<= steps*subSteps; i++){
		start = MPI_Wtime();
		
		// turn array of doubles into array of vec3 objects
		position = convertArray(positionD, total*3);
		
		// calculate first and last index of broadcasted arrays to be worked on
		int first = offset[my_rank] / 3;
		int last = first + portion;

		// calculate the force and corresponding new positions and velocities using OpenMP for optimization
#pragma omp parallel default(none) shared (position,force, mass, total, timeSteps, height, width, depth, portion, my_position, velocity, first, last, my_rank) num_threads(4)
		{
			calculateForce (position, force, mass, total, first, last);
			calculateStep(my_position, velocity, force, mass, timeSteps, first, (portion), height, width, depth);
			
		}

		// translate new positions back into an array of doubles and update each node with the new values
		double *my_positionD = &my_position[0].x;
		MPI_Allgatherv (my_positionD, (portion) * 3, MPI_DOUBLE, positionD, distribution, offset, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Barrier (MPI_COMM_WORLD);
		end = MPI_Wtime();

		// root node generates the image
		if(my_rank == 0){
	
			// initialize variables to be used in this part
			image = (unsigned char *) malloc (sizeof (unsigned char) * 3 * (width*height));
			int j,k, l, minY, maxY, minX, maxX, radius;
			int radiusF = 5; // max size of a particle

			// similar to assignment 3, paint a square for each particle based on it's position and the boundaries of the image
			for (j = 0; j < light; j++){
				radius = (int)(radiusF * (1-(position[j].z / depth)));
				if (position[j].y < height && position[j].x < width){
					minY = position[j].y - radius < 0;
					minY = (1-minY)*(position[j].y-radius);
					maxY = position[j].y + radius < height;
					maxY = maxY * (position[j].y +radius)+((1-maxY)*height);
					minX = position[j].x -radius< 0;
					minX = (1-minX) * (position[j].x - radius);
					maxX = position[j].x + radius < width;
					maxX = maxX*(radius +position[j].x) + ((1-maxX)*width);
					for (k = minY; k < maxY; k++){
						for (l = minX; l < maxX; l++){
							int index = ((k*width)+l)*3;
							image[index] = 255;
						}
					}  
				}
			}

			for (j = light; j < light+medium; j++){
				radius = (int)(radiusF * (1-(position[j].z / depth)));
				if (position[j].y < height && position[j].x < width){
					minY = position[j].y - radius < 0;
					minY = (1-minY)*(position[j].y-radius);
					maxY = position[j].y + radius < height;
					maxY = maxY * (position[j].y +radius)+((1-maxY)*height);
					minX = position[j].x -radius< 0;
					minX = (1-minX) * (position[j].x - radius);
					maxX = position[j].x + radius < width;
					maxX = maxX*(radius +position[j].x) + ((1-maxX)*width);
					for (k = minY; k < maxY; k++){
						for (l = minX; l < maxX; l++){
							int index = ((k*width)+l)*3;
							image[index+1] = 255;
						}
					}
				}
			}

			for (j = light+medium; j < total;j++){
				radius = (int)(radiusF * (1-(position[j].z / depth)));
				if (position[j].y < height && position[j].x < width){
					minY = position[j].y - radius < 0;
					minY = (1-minY)*(position[j].y-radius);
					maxY = position[j].y + radius < height;
					maxY = maxY * (position[j].y +radius)+((1-maxY)*height);
					minX = position[j].x -radius < 0;
					minX = (1-minX) * (position[j].x - radius);
					maxX = position[j].x + radius < width;
					maxX = maxX*(radius +position[j].x) + ((1-maxX)*width);
					for (k = minY; k < maxY; k++){
						for (l = minX; l < maxX; l++){
							int index = ((k*width)+l)*3;
							image[index+2] = 255;
						}
					}
				}
			}

			// save the image based on the updated image array
			if (i % subSteps == 0){

				// construct filename which is the imageFilenamePrefix + iamge number(padded with leading 0s so that it's 6 digits long).bmp
				int logVal = (int)(log10(count));
				char *imageNum = (char *)malloc (sizeof(char) * 5);
				char *filename = (char *) malloc (sizeof (char) * sizeof(argv[9]));
				char *zeros = (char *) malloc (sizeof (char) * 5);
				strcpy (filename, argv[9]);
				int n;
				n = sprintf(zeros, "%d", 0);
				for (int p = 0; p < 5-(logVal + 1); p++){
					strcat (zeros, "0");
				}
				n = sprintf(imageNum,"%s%d",zeros,count);
				(void) n;
				strcat (imageNum, ".bmp");

				// save the image and update the runtimes array
				saveBMP(strcat(filename, imageNum), image, width, height);
				runtimes[count-1] = end-start;
				count++;
			}
		}
	}
	// free up the memory allocated for all the arrays and finalize MPI stuff
	free(image);
	free(offset);
	free(distribution);
	free(force);
	free(velocity);
	free(position);
	free(mass);
	free(my_velocity);
	free(my_position);
	MPI_Finalize();

	// root processor finds the minimum, maximum, and average step time
	if (my_rank == 0 ){
		int z;
		double max = 0.0;
		double min = INFINITY;
		double sum = 0;
		for (z = 0; z < count-1; z++){
			if (runtimes[z] > max){
				max = runtimes[z];
			}
			if (runtimes[z] < min){
				min = runtimes[z];
			}
			sum += runtimes[z];
		}
		printf("%f %f %f\n", min, max, sum/(count-1));
	}
	free(runtimes);
	return 0;
}

