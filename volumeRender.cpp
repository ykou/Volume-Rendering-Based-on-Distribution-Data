/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include <iostream>
#include <cstdio>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

//const char *volumeFilename = "Bucky.raw";
const char *volumeFilename = "isabel_blocknumber_50_50_10_bins_32_withoutBound.bin";
const char *codebookFilename = "isabelqvp_codebook0_50x50x10blocks_32bins.bin";
const char *templatesFilename = "isabelqvp_domainList0_50x50x10blocks_32bins.bin";
const char *spanListFilename = "spanList.bin";
const char *fractalHistogramFilename = "codebook0.bin";	//this is for flexible size fractal
const char *simpleHistogramCountFilename = "nzbCounts0.bin";
const char *simpleHistogramBinIdFilename = "nzbBinIds0.bin";
const char *simpleHistogramBinFreqFilename = "nzbFreqs0.bin";
const char *flexibleTemplatesFilename = "domainList.bin";

cudaExtent volumeSize = make_cudaExtent(50, 50, 10);
cudaExtent histogramSize = make_cudaExtent(32, (int)50*50, 10);
cudaExtent codebookSize = make_cudaExtent(50, 50, 10);	//TODO: confirm that the size is 50x50x10
cudaExtent templatesSize = make_cudaExtent(32, 622, 1);		//TODO: change the size if we change the template file
cudaExtent errorsbookSize = make_cudaExtent(32, (int)50*50, 10);
//typedef unsigned char VolumeType;
typedef float VolumeType;
//typedef int BoundType;
const int nBlocks = (int) 50 * 50 * 10;
const int nBins = 32;
const int flexibleNBins = 64;
typedef int4 SpanType;

int4 *codebook;
float2 *errorsbook;
float *templates;	// NOTE: it seems that CUDA only support double1 instead of double, we may try it later

// flexible block size
SpanType *spanLow;
SpanType *spanHigh;
SpanType *codebookSpanLow;		// Note: here we only record the span list for fractal encoded histograms, and make sure span and codebook have the same index
SpanType *codebookSpanHigh;
int4 *flexibleCodebook;
float2 *flexibleErrorsbook;
SpanType *simpleLow;
SpanType *simpleHigh;
int *simpleCount;
float2 *simpleHistogram;
float *flexibleTemplates;


//char *volumeFilename = "mrt16_angio.raw";
//cudaExtent volumeSize = make_cudaExtent(416, 512, 112);
//typedef unsigned short VolumeType;

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

int queryMethod = 1;	//1 is mean, 2 is variance, 3 is entropy, 4 is fractal mean, 5 is fractal variance, 6 is fractal entropy, 7 is interpolated mean, 8 flexBlockSize entropy, 9 flexBlockSize mean, 0 flexBlockSize variance
float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize,
		cudaExtent histogramSize, int4 *h_codebook, cudaExtent codebookSize,
		float *h_templates, cudaExtent templatesSize, float2 *h_errorsbook,
		cudaExtent errorsbookSize, SpanType *h_codebookSpanLow,
		SpanType *h_codebookSpanHigh, int4 *h_flexibleCodebook,
		float2 *h_flexibleErrorsbook, SpanType *h_simpleLow, SpanType *h_simpleHigh,
		int *h_simpleCount, float2 *h_simpleHistogram, float *h_flexibleTemplates);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, int queryMethod, cudaExtent volumeSize);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

extern "C" void dataProcessing();
extern "C" void basicDataProcessing();

void initPixelBuffer();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Volume Render: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

// render image using CUDA
void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
	//printf("volumeSize: %d %d %d\n", volumeSize.width, volumeSize.height, volumeSize.depth);
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, queryMethod, volumeSize);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            exit(EXIT_SUCCESS);
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;

        case '+':
            density += 0.01f;
            break;

        case '-':
            density -= 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;

        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;

        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;

        case ',':
            transferScale -= 0.01f;
            break;

		case '1':
			queryMethod = 1;
			break;
		case '2':
			queryMethod = 2;
			break;
		case '3':
			queryMethod = 3;
			break;
		case '4':
			queryMethod = 4;
			break;
		case '5':
			queryMethod = 5;
			break;
		case '6':
			queryMethod = 6;
			break;
		case '7':
			queryMethod = 7;
			break;
		case '8':
			queryMethod = 8;
			break;
		case '9':
			queryMethod = 9;
			break;
		case '0':
			queryMethod = 0;
			break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b) ;
}

void reshape(int w, int h)
{
    width = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
//void *loadRawFile(char *filename, size_t size)
//{
//    FILE *fp = fopen(filename, "rb");
//
//    if (!fp)
//    {
//        fprintf(stderr, "Error opening file '%s'\n", filename);
//        return 0;
//    }
//
//    void *data = malloc(size);
//    size_t read = fread(data, 1, size, fp);
//    fclose(fp);
//
//    printf("Read '%s', %d bytes\n", filename, read);
//
//    return data;
//}
float *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

//    float *data = malloc(size);
    float *data = new float[nBlocks * nBins];
    size_t read = fread((void*)data, sizeof(float), nBlocks * nBins, fp);
    fclose(fp);

    printf("Read '%s', %d float\n", filename, read);

    return data;
}

int loadCodebook(char *filename) {


    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    // <number of time steps - INT>
    int *nSteps = new int;
    size_t read_nStep = fread((void*)nSteps, sizeof(int), 1, fp);
    printf("nStep: %d\n", *nSteps);

    // <number of blocks - INT>
    int *nBlocks = new int;
    size_t read_nBlocks = fread((void*)nBlocks, sizeof(int), 1, fp);
    printf("nBlocks: %d\n", *nBlocks);

    // the codebook data that will be tranfered to GPU
    codebook = new int4[*nBlocks];
   // the errorsbook data that will be tranfered to GPU
	errorsbook = new float2[*nBlocks * nBins];

    for (int i = 0; i < *nBlocks; i++) {
        // <spanId - INT> -- ignore this
        int *spanId = new int;
        size_t read_spanId = fread((void*)spanId, sizeof(int), 1, fp);
        // printf("spanId: %d\n", *spanId);

        // <template id - INT>
        int *templateId = new int;
        size_t read_templateId = fread((void*)templateId, sizeof(int), 1, fp);
        //if (*templateId != 32) {
		//	printf("template ID: %d\n", *templateId);
		//}

        // <amount of shift - INT>
        int *shift = new int;
        size_t read_shift = fread((void*)shift, sizeof(int), 1, fp);
        //printf("amount of shift: %d\n", *shift);

        // <reflection flag> -- this is a boolean
        bool *reflectionFlag = new bool;
        size_t read_relectionFlag = fread((void*)reflectionFlag, sizeof(bool), 1, fp);
        // printf("reflection flag: %d\n", *reflectionFlag);

        // <number or errors stored, say NE - INT>
        int *nErrors = new int;
        size_t read_nErrors = fread((void*)nErrors, sizeof(int), 1, fp);
        // printf("number of errors: %d\n", *nErrors);
		if (*nErrors > nBins) {
			// printf("nErrors > nBins !\n");
			return 0;
		}

        // put templateId, amoutOfShift, reflectionFlag and number of errors into codebook
        codebook[i] = make_int4(*templateId, *shift, *reflectionFlag, *nErrors);

        // <a list of NE integers, storing the bin Ids where errors are>
        int *NEbinIds = new int[*nErrors];
        size_t read_NEbinIds = fread((void*)NEbinIds, sizeof(int), *nErrors, fp);
        // printf("NEbinIds: \n");
        for (int j = 0; j < *nErrors; j++) {
            // printf("%d ", NEbinIds[j]);
			errorsbook[(int)i * nBins + j].x = (float) NEbinIds[j];
        }
        // printf("\n");

        // <a list of NE doubles, storing the corresponding error>
        double *NEvalues = new double[*nErrors];
        size_t read_NEvalues = fread((void*)NEvalues, sizeof(double), *nErrors, fp);
        // printf("NEvalues: \n");
        for (int k = 0; k < *nErrors; k++) {
            // printf("%f ", NEvalues[k]);
			errorsbook[(int)i * nBins + k].y = (float) NEvalues[k];
        }
        // printf("\n");
    }

	fclose(fp);
    return 1;
}

int loadTemplates(char *filename) {



    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    // <number of blocks - INT>
    int *nTemplates = new int;
    size_t read_nTemplates = fread((void*)nTemplates, sizeof(int), 1, fp);
    printf("nTemplates: %d\n", *nTemplates);

	templates = new float[(int)((*nTemplates) * nBins)];

    for (int i = 0; i < *nTemplates; i++) {
        // <limits - 6 DOUBLES> -- ignore this
        double *nLimits = new double[6];
        size_t read_nLimits = fread((void*)nLimits, sizeof(double), 6, fp);
        // printf("nLimits: \n");
        for (int j = 0; j < 6; j++) {
            // printf("%f ", nLimits[j]);
        }
        // printf("\n");

        // <list of nBin doubles which is the frequency list>
        double *nFrequencies = new double[nBins];
        size_t read_nFrequencies = fread((void*)nFrequencies, sizeof(double), nBins, fp);
        // printf("nFrequencies: \n");
        for (int k = 0; k < nBins; k++) {
            if (nFrequencies[k] < 0 || nFrequencies[k] > 1) {
               printf("Error! nFrequencies[%d]: %f\n", k, nFrequencies[k]);
            }
			templates[(int)(i * nBins + k)] = (float) nFrequencies[k];
			if (templates[(int)(i * nBins + k)] < 0 || templates[(int)(i * nBins + k)] > 1) {
				printf("Error! templates(%d, %d): %f\n", i, k, templates[(int)(i * nBins + k)]);
			}
        }
        // printf("\n");
    }

	fclose(fp);
    return 1;
}

int checkSpanLimit(SpanType low, SpanType high) {
	if (low.x > high.x || low.y > high.y || low.z > high.z || low.x < 0 || low.y < 0 || low.z < 0 || high.x < 0 || high.y < 0 || high.z < 0) {
	//if (low.x < 0 || low.y < 0 || low.z < 0 || high.x < 0 || high.y < 0 || high.z < 0) {
		return 0;
	}
	return 1;
}

int checkHistogram(float2 histogram) {
	if (histogram.x < 0 || histogram.y < 0 || histogram.x > flexibleNBins || histogram.y > 1.0) {
		printf("Wrong histogram!\n");
		return 0;
	}
	return 1;
}

int loadSpanList(char *filename) {
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    int nSpan = 0;
    size_t read_nSpan = fread((void*)&nSpan, sizeof(int), 1, fp);
    printf("nSpan: %d\n", nSpan);

    spanLow = new SpanType[nSpan];
    spanHigh = new SpanType[nSpan];

	int3 MinLow = make_int3(64, 64, 64);
	int3 MaxHigh = make_int3(0, 0, 0);
    for (int i = 0; i < nSpan; i++) {
    	int lowX = -1;
    	int lowY = -1;
    	int lowZ = -1;
    	int highX = -1;
    	int highY = -1;
    	int highZ = -1;
    	size_t read_lowX = fread((void*)&lowX, sizeof(int), 1, fp);
    	size_t read_lowY = fread((void*)&highX, sizeof(int), 1, fp);
    	size_t read_lowZ = fread((void*)&lowY, sizeof(int), 1, fp);
    	size_t read_highX = fread((void*)&highY, sizeof(int), 1, fp);
    	size_t read_highY = fread((void*)&lowZ, sizeof(int), 1, fp);
    	size_t read_highZ = fread((void*)&highZ, sizeof(int), 1, fp);
    	spanLow[i] = make_int4(lowX, lowY, lowZ, 0);
    	spanHigh[i] = make_int4(highX, highY, highZ, 0);
    	if (checkSpanLimit(spanLow[i], spanHigh[i]) != 1) {
			printf("Wrong Low or High! Low: %d %d %d, High: %d %d %d\n", spanLow[i].x, spanLow[i].y, spanLow[i].z, spanHigh[i].x, spanHigh[i].y, spanHigh[i].z);
    		return 0;
    	}
		if (MinLow.x > lowX) { MinLow.x = lowX; }
		if (MinLow.y > lowY) { MinLow.y = lowY; }
		if (MinLow.z > lowZ) { MinLow.z = lowZ; }
		if (MinLow.x > highX) { MinLow.x = highX; }
		if (MinLow.y > highY) { MinLow.y = highY; }
		if (MinLow.z > highZ) { MinLow.z = highZ; }

		if (MaxHigh.x < lowX) { MaxHigh.x = lowX; }
		if (MaxHigh.y < lowY) { MaxHigh.y = lowY; }
		if (MaxHigh.z < lowZ) { MaxHigh.z = lowZ; }
		if (MaxHigh.x < highX) { MaxHigh.x = highX; }
		if (MaxHigh.y < highY) { MaxHigh.y = highY; }
		if (MaxHigh.z < highZ) { MaxHigh.z = highZ; }
		//if (spanLow[i].x > spanHigh[i].x || spanLow[i].y > spanHigh[i].y || spanLow[i].z > spanHigh[i].z) {
		//	printf("Low: %d %d %d, High: %d %d %d\n", spanLow[i].x, spanLow[i].y, spanLow[i].z, spanHigh[i].x, spanHigh[i].y, spanHigh[i].z);
		//}

		//if (spanLow[i].x != 0 && spanLow[i].y != 0 && spanLow[i].z != 0) {
		//	printf("index: %6d, Low: %2d %2d %2d, High: %2d %2d %2d\n", i, spanLow[i].x, spanLow[i].y, spanLow[i].z, spanHigh[i].x, spanHigh[i].y, spanHigh[i].z);
		//}
    }
	//printf("MinLow: %d, %d, %d, MaxHigh: %d, %d, %d\n", MinLow.x, MinLow.y, MinLow.z, MaxHigh.x, MaxHigh.y, MaxHigh.z);

    fclose(fp);
	return 1;
}

int loadFractalHistogram(char *filename) {
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    int nTimeSteps = 0;
    int nHistogram = 0;
    size_t read_nTimeStep = fread((void*)&nTimeSteps, sizeof(int), 1, fp);
    size_t read_nHistogram = fread((void*)&nHistogram, sizeof(int), 1, fp);
    printf("Fractal nHistogram: %d\n", nHistogram);

    flexibleCodebook = new int4[nHistogram];
	flexibleErrorsbook = new float2[nHistogram * flexibleNBins];
	codebookSpanLow = new SpanType[nHistogram];
	codebookSpanHigh = new SpanType[nHistogram];

	int countZero = 0; // count how many (0, 0, 0, 0) fractal histogram
    for (int i = 0; i < nHistogram; i++) {

        // <spanId - INT> -- ignore this
        int spanId = -1;
        size_t read_spanId = fread((void*)&spanId, sizeof(int), 1, fp);
		//printf("spanId: %d\n", spanId);
        if (spanId < 0 || spanId > 2 * nHistogram) {
        	printf("Wrong spanId! %d\n", spanId);
        	return 0;
        }


        // <template id - INT>
        int templateId = -1;
        size_t read_templateId = fread((void*)&templateId, sizeof(int), 1, fp);
		//printf("templateId: %d\n", templateId);
        if (templateId < 0) {
			printf("Wrong template ID: %d\n", templateId);
			return 0;
		}

        // <amount of shift - INT>
        int shift = -1;
        size_t read_shift = fread((void*)&shift, sizeof(int), 1, fp);
		//printf("shift: %d\n", shift);
        if (shift < 0) {
        	printf("Wrong amount of shift: %d\n", shift);
        }

        // <reflection flag> -- this is a boolean
        bool reflectionFlag = false;
        size_t read_relectionFlag = fread((void*)&reflectionFlag, sizeof(bool), 1, fp);
        //printf("reflection: %d\n", reflectionFlag);

        // <number or errors stored, say NE - INT>
        int nErrors = -1;
        size_t read_nErrors = fread((void*)&nErrors, sizeof(int), 1, fp);
		//printf("nErrors: %d\n", nErrors);
        // printf("number of errors: %d\n", *nErrors);
		if (nErrors > flexibleNBins || nErrors < 0) {
			printf("Wrong! nErrors > flexibleNBins or nError <0:  %d!\n", nErrors);
			return 0;
		}

        // put templateId, amoutOfShift, reflectionFlag and number of errors into codebook
		flexibleCodebook[i] = make_int4(templateId, shift, reflectionFlag, nErrors);
		codebookSpanLow[i] = spanLow[spanId];
		codebookSpanHigh[i] = spanHigh[spanId];

		//printf("%6d, spanId %d, Id: %3d, shift %d, flip %d, nErrors %d\n", i, spanId, templateId, shift, reflectionFlag, nErrors);
		if (templateId == 0 && shift == 0 && reflectionFlag == 0) {
			countZero++;
			//printf("low(%2d, %2d, %2d), high(%2d, %2d, %2d), i %d, template %3d, shift %2d, flip %d, NE %d\n", codebookSpanLow[i].x, codebookSpanLow[i].y, codebookSpanLow[i].z, codebookSpanHigh[i].x, codebookSpanHigh[i].y, codebookSpanHigh[i].z, i, flexibleCodebook[i].x, flexibleCodebook[i].y, flexibleCodebook[i].z, flexibleCodebook[i].w);
		}

        // <a list of NE integers, storing the bin Ids where errors are>
        int *NEbinIds = new int[nErrors];
        size_t read_NEbinIds = fread((void*)NEbinIds, sizeof(int), nErrors, fp);
        // printf("NEbinIds: \n");
        for (int j = 0; j < nErrors; j++) {
            // printf("%d ", NEbinIds[j]);
			flexibleErrorsbook[(int)i * flexibleNBins + j].x = (float) NEbinIds[j];
        }
        // printf("\n");

        // <a list of NE doubles, storing the corresponding error>
        double *NEvalues = new double[nErrors];
        size_t read_NEvalues = fread((void*)NEvalues, sizeof(double), nErrors, fp);
        // printf("NEvalues: \n");
        for (int k = 0; k < nErrors; k++) {
            // printf("%f ", NEvalues[k]);
			flexibleErrorsbook[(int)i * flexibleNBins + k].y = (float) NEvalues[k];
        }
        // printf("\n");
		//printf("here! i = %d\n", i);
    }
	//printf("finish loop!\n");
	printf("countZero: %d\n", countZero);
    fclose(fp);

	return 1;
}

int loadSimpleHistogram(char *binCountFilename, char *binIdFilename, char *binFreqFilename) {
    FILE *fp = fopen(binCountFilename, "rb");
    FILE *fpBinId = fopen(binIdFilename, "rb");
    FILE *fpBinFreq = fopen(binFreqFilename, "rb");

    if ( (!fp) || (!fpBinId) || (!fpBinFreq) )
    {
        fprintf(stderr, "Error opening file '%s'\n", binCountFilename);
        fprintf(stderr, "Error opening file '%s'\n", binIdFilename);
        fprintf(stderr, "Error opening file '%s'\n", binFreqFilename);
        return 0;
    }

    int nHistogram = -1;
    size_t read_nHistogram = fread((void*)&nHistogram, sizeof(int), 1, fp);
    printf("Simple nHistogram: %d\n", nHistogram);

    simpleCount = new int[nHistogram];
    simpleLow = new int4[nHistogram];
    simpleHigh = new int4[nHistogram];
    simpleHistogram = new float2[nHistogram * flexibleNBins];

    for (int i = 0; i < nHistogram; i++) {
    	simpleLow[i] = make_int4(-1, -1, -1, 0);
    	simpleHigh[i] = make_int4(-1, -1, -1, 0);
    	size_t read_lowX = fread((void*)&(simpleLow[i].x), sizeof(int), 1, fp);
    	size_t read_lowY = fread((void*)&(simpleLow[i].y), sizeof(int), 1, fp);
    	size_t read_lowZ = fread((void*)&(simpleLow[i].z), sizeof(int), 1, fp);
    	size_t read_highX = fread((void*)&(simpleHigh[i].x), sizeof(int), 1, fp);
    	size_t read_highY = fread((void*)&(simpleHigh[i].y), sizeof(int), 1, fp);
    	size_t read_highZ = fread((void*)&(simpleHigh[i].z), sizeof(int), 1, fp);
		if (checkSpanLimit(simpleLow[i], simpleLow[i]) != 1) {
			printf("Wrong simpleLow or simpleHigh!\n");
			return 0;
		}
		if (simpleLow[i].x == 64 || simpleLow[i].y == 64 || simpleLow[i].z == 64 || simpleHigh[i].x == 64 || simpleHigh[i].y == 64 || simpleHigh[i].z == 64) {
			printf("%d, spanLow(%2d, %2d, %2d), spanHigh(%2d, %2d, %2d)\n", i, simpleLow[i].x, simpleLow[i].y, simpleLow[i].z, simpleHigh[i].x, simpleHigh[i].y, simpleHigh[i].z);
		}

		// read number of non-zero bins
		simpleCount[i] = -1;
		size_t read_count = fread((void*)&(simpleCount[i]), sizeof(int), 1, fp);
		if (simpleCount[i] < 0) {
			printf("Wrong simple Count: %d\n", simpleCount[i]);
			return 0;
		}


		// read binId and binFreq
		float total = 0;
		for (int j = 0; j < simpleCount[i]; j++) {
			int binId = -1;
			double binFreq = -1.0;
			size_t read_binId = fread((void*)&binId, sizeof(int), 1, fpBinId);
			size_t read_binFreq = fread((void*)&binFreq, sizeof(double), 1, fpBinFreq);
			simpleHistogram[i * flexibleNBins + j] = make_float2((float)binId, (float)binFreq);
			if (checkHistogram(simpleHistogram[i * flexibleNBins + j]) != 1) {
				printf("Wrong simple histogram! count %d, binId %d, binFreq %f\n", simpleCount[i], binId, binFreq);
				return 0;
			}
			total += binFreq;
			// test if total is 1.0
		}
		if (total > 1.000001 || total < 0.999999) {
			printf("simple i = %6d, total = %f\n", i, total);
		}
    }

    fclose(fp);
    fclose(fpBinId);
    fclose(fpBinFreq);
	return 1;
}

int loadFlexibleTemplates(char *filename) {

    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    // <number of blocks - INT>
    int nTemplates = -1;
    int nTemplatesUsed = -1;
    size_t read_nTemplates = fread((void*)&nTemplates, sizeof(int), 1, fp);
    //size_t read_nTemplatesUsed = fread((void*)&nTemplatesUsed, sizeof(int), 1, fp);
    printf("nTemplates: %d\n", nTemplates);

	flexibleTemplates = new float[nTemplates * flexibleNBins];

    for (int i = 0; i < nTemplates; i++) {
        // <limits - 6 DOUBLES> -- ignore this
        double *nLimits = new double[6];
        size_t read_nLimits = fread((void*)nLimits, sizeof(double), 6, fp);
        // printf("nLimits: \n");
        //for (int j = 0; j < 6; j++) {
        //    printf("%f ", nLimits[j]);
        //}
        // printf("\n");

        // <list of nBin doubles which is the frequency list>
        double *nFrequencies = new double[flexibleNBins];
        size_t read_nFrequencies = fread((void*)nFrequencies, sizeof(double), flexibleNBins, fp);
        // printf("nFrequencies: \n");
        for (int k = 0; k < flexibleNBins; k++) {
            flexibleTemplates[(int)(i * flexibleNBins + k)] = (float) nFrequencies[k];
			if (flexibleTemplates[(int)(i * flexibleNBins + k)] < 0 || flexibleTemplates[(int)(i * flexibleNBins + k)] > 1) {
			//if (flexibleTemplates[(int)(i * flexibleNBins + k)] != 0) {
				printf("Error! flexibleTemplates(%d, %d): %f\n", i, k, flexibleTemplates[(int)(i * flexibleNBins + k)]);
				return 0;
			}
        }
        // printf("\n");
    }

	fclose(fp);
    return 1;
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    int result = 0;

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, argv);
    }
    else
    {
        result = findCudaDevice(argc, argv);
    }

    return result;
}

void runSingleTest(const char *ref_file, const char *exec_path)
{
    bool bTestResult = true;

    uint *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

    float modelView[16] =
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 4.0f, 1.0f
    };

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    // call CUDA kernel, writing results to PBO
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // Start timer 0 and process n loops on the GPU
    int nIter = 10;

    for (int i = -1; i < nIter; i++)
    {
        if (i == 0)
        {
            cudaDeviceSynchronize();
            sdkStartTimer(&timer);
        }

        render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, queryMethod, volumeSize);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = sdkGetTimerValue(&timer)/(nIter * 1000.0);
    printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


    getLastCudaError("Error: render_kernel() execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char *h_output = (unsigned char *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

    sdkSavePPM4ub("volume.ppm", h_output, width, height);
    bTestResult = sdkComparePPM("volume.ppm", sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, true);

    cudaFree(d_output);
    free(h_output);
    cleanup();

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    char *ref_file = NULL;

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
        fpsLimit = frameCheckNumber;
    }

    if (ref_file)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, false);
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, true);
    }

    // parse arguments
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "volume", &filename))
    {
        volumeFilename = filename;
    }

    int n;

    if (checkCmdLineFlag(argc, (const char **) argv, "size"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "size");
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "xsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "xsize");
        volumeSize.width = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "ysize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "ysize");
        volumeSize.height = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "zsize"))
    {
        n= getCmdLineArgumentInt(argc, (const char **) argv, "zsize");
        volumeSize.depth = n;
    }

    // load volume data
    char *path = sdkFindFilePath(volumeFilename, argv[0]);

    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

	char *codebookPath = sdkFindFilePath(codebookFilename, argv[0]);
	char *templatesPath = sdkFindFilePath(templatesFilename, argv[0]);
	char *spanListPath = sdkFindFilePath(spanListFilename, argv[0]);
	char *fractalHistogramPath = sdkFindFilePath(fractalHistogramFilename, argv[0]);
	char *simpleHistogramCountPath = sdkFindFilePath(simpleHistogramCountFilename, argv[0]);
	char *simpleHistogramBinIdPath = sdkFindFilePath(simpleHistogramBinIdFilename, argv[0]);
	char *simpleHistogramBinFreqPath = sdkFindFilePath(simpleHistogramBinFreqFilename, argv[0]);
	char *flexibleTemplatesPath = sdkFindFilePath(flexibleTemplatesFilename, argv[0]);

	if (codebookPath == 0 || templatesPath == 0) {
		printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*(sizeof(VolumeType) * nBins);

    float *h_volume = loadRawFile(path, size);

	int result_readCodebook = loadCodebook(codebookPath);
	int result_readTemplates = loadTemplates(templatesPath);
	int result_readSpanList = loadSpanList(spanListPath);
	int result_readFractal = loadFractalHistogram(fractalHistogramPath);
	int result_readSimple = loadSimpleHistogram(simpleHistogramCountPath, simpleHistogramBinIdPath, simpleHistogramBinFreqPath);
	int result_readFlexibleTemplates = loadFlexibleTemplates(flexibleTemplatesPath);

	if (result_readCodebook == 0 || result_readTemplates == 0 || result_readSpanList == 0 || result_readFractal == 0 || result_readSimple == 0 || result_readFlexibleTemplates == 0) {
		printf("Wrong Codebook or Templates!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 64 * 64 * 32; i++) {
		if (simpleLow[i].x == 64 || simpleLow[i].y == 64 || simpleLow[i].z == 64 || simpleHigh[i].x == 64 || simpleHigh[i].y == 64 || simpleHigh[i].z == 64) {
			printf("%2d %2d %2d, %2d %2d %2d\n", simpleLow[i].x, simpleLow[i].y, simpleLow[i].z, simpleHigh[i].x, simpleHigh[i].y, simpleHigh[i].z);
		}
	}

    initCuda(h_volume, volumeSize, histogramSize,
    		codebook, codebookSize, templates, templatesSize, errorsbook, errorsbookSize,
    		codebookSpanLow, codebookSpanHigh, flexibleCodebook, flexibleErrorsbook,
    		simpleLow, simpleHigh, simpleCount, simpleHistogram, flexibleTemplates);

    free(h_volume);
	// free all pointers
	free(codebook);
	free(templates);
	free(errorsbook);
	free(codebookSpanLow);
	free(codebookSpanHigh);
	free(flexibleCodebook);
	free(flexibleErrorsbook);
	free(simpleLow);
	free(simpleHigh);
	free(simpleCount);
	free(simpleHistogram);
	free(flexibleTemplates);

	dataProcessing();
	basicDataProcessing();

    sdkCreateTimer(&timer);

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	std::cout << "GridSize: " << gridSize.x << ", " << gridSize.y << ", " << gridSize.z  << std::endl;

    if (ref_file)
    {
        runSingleTest(ref_file, argv[0]);
    }
    else
    {
        // This is the normal rendering path for VolumeRender
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutReshapeFunc(reshape);
        glutIdleFunc(idle);

        initPixelBuffer();

        atexit(cleanup);

        glutMainLoop();
    }

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
