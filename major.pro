
QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Major
DESTDIR = ./bin
TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++11
CONFIG += c++11

SOURCES += $$PWD/src/*.cpp              \
            $$PWD/src/MeshSampler/*.cpp \
            $$PWD/src/Mesh/*.cpp        \
            $$PWD/src/FluidSystem/*.cpp \
            $$PWD/src/SPH/*.cpp         \
            $$PWD/src/Render/*.cpp      \
            $$PWD/src/Widget/*.cpp

HEADERS  += $$PWD/include/*.h               \
            $$PWD/include/MeshSampler/*.h   \
            $$PWD/include/Mesh/*.h          \
            $$PWD/include/FluidSystem/*.h   \
            $$PWD/include/SPH/*.h           \
            $$PWD/include/Render/*.h        \
            $$PWD/include/Widget/*.h

OTHER_FILES +=  shader/* \
                shader/Fluid/* \
                shader/Skybox/*

INCLUDEPATH +=  $$PWD/include \
                /usr/local/include \
                /usr/include \
                /home/idris/dev/include

LIBS += -L/usr/local/lib -L/usr/lib -lGL -lGLU -lGLEW \
        -L/home/idris/dev/lib -L/usr/local/lib -lassimp

OBJECTS_DIR = ./obj
MOC_DIR = ./moc
FORMS    += ./form/mainwindow.ui

UI_DIR += ./ui


#--------------------------------------------------------------------------
# CUDA stuff
#--------------------------------------------------------------------------

HEADERS += $$PWD/cuda_inc/*.cuh

INCLUDEPATH += $$PWD/cuda_inc
CUDA_SOURCES += $$PWD/cuda_src/*.cu
CUDA_PATH = /usr
NVCC = $$CUDA_PATH/bin/nvcc

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
GENCODE_FLAGS += -arch=sm_50
NVCC_OPTIONS =  -ccbin g++ --use_fast_math -std=c++11 #-dc

# include paths
INCLUDEPATH += $(CUDA_PATH)/include $(CUDA_PATH)/include/cuda

# library directories
QMAKE_LIBDIR += $$CUDA_PATH/lib/x86_64-linux-gnu $(CUDA_PATH)/include/cuda

CUDA_OBJECTS_DIR = $$PWD/cuda_obj

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,' -I','-I','')
LIBS += -lcudart -lcurand #-lcudadevrt

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$NVCC -m$$SYSTEM_TYPE $$GENCODE_FLAGS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} $$NVCC_OPTIONS $$CUDA_INC #--relocatable-device-code=true --compile
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda




#--------------------------------------------------------------------------
# CUDA dynamic stuff?
#--------------------------------------------------------------------------

##set out cuda sources
#CUDA_SOURCES += ./cuda_src/*.cu
#HEADERS += $$PWD/cuda_inc/*.cuh

## Path to cuda SDK install
#linux:CUDA_DIR = /usr

##Cuda include paths
#INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
#INCLUDEPATH += $$PWD/cuda_inc


##cuda libs
#macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
#linux:QMAKE_LIBDIR += $$CUDA_DIR/lib
#QMAKE_LIBDIR += $$CUDA_DIR/lib
#QMAKE_LIBDIR += $$CUDA_DIR/lib/x86_64-linux-gnu
#QMAKE_LIBDIR += $(CUDA_PATH)/include/cuda
#LIBS += -L$$CUDA_DIR/lib -L$$CUDA_DIR/lib/x86_64-linux-gnu -lcudart -lcudadevrt

## join the includes in a line
#CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')


## nvcc flags (ptxas option verbose is always useful)
#NVCCFLAGS = -dc --compiler-options -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20


##prepare intermediat cuda compiler
#cudaIntr.input = CUDA_SOURCES
#cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o

### Tweak arch according to your hw's compute capability
#cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode arch=compute_50,code=sm_50 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

##Set our variable out. These obj files need to be used to create the link obj file
##and used in our final gcc compilation
#cudaIntr.variable_out = CUDA_OBJ
#cudaIntr.variable_out += OBJECTS
#cudaIntr.clean = cudaIntrObj/*.o

#QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr


## Prepare the linking compiler step
#cuda.input = CUDA_OBJ
#cuda.output = ${QMAKE_FILE_BASE}_link.o

## Tweak arch according to your hw's compute capability
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode arch=compute_50,code=sm_50 $$LIBS -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
## Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_UNIX_COMPILERS += cuda


