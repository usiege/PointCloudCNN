#include <stdio.h>
#include <stdlib.h>
#include "matio.h"


int main(int argc,char **argv)
{
    mat_t *matfp;
    matvar_t *matvar, *field;
    size_t dims[2] = {10,1}, struct_dims[2] = {2,1};
    double x1[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9,10},
    x2[10] = {11,12,13,14,15,16,17,18,19,20},
    y1[10] = {21,22,23,24,25,26,27,28,29,30},
    y2[10] = {31,32,33,34,35,36,37,38,39,40};
    struct mat_complex_split_t z1 = {x1,y1}, z2 = {x2,y2};
    const char *fieldnames[3] = {"x","y","z"};
    unsigned nfields = 3;

    matfp = Mat_CreateVer("/media/charles/exFAT/working/PointCloud/scripts/dgcnn_lua/data/raw_data/MUTAG.mat",
        NULL,MAT_FT_DEFAULT);
    if ( NULL == matfp ) 
    {
        fprintf(stderr,"Error creating MAT file \"test.mat\"\n");
        return EXIT_FAILURE;
    }
    matvar = Mat_VarCreateStruct("a", 2,struct_dims,fieldnames,nfields);
    if ( NULL == matvar ) 
    {
        fprintf(stderr,"Error creating variable for ’a’\n");
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    /* structure index 0 */
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,x1,0);
    Mat_VarSetStructFieldByName(matvar,"x",0,field);
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,y1,0);
    Mat_VarSetStructFieldByName(matvar,"y",0,field);
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,&z1,MAT_F_COMPLEX);
    Mat_VarSetStructFieldByName(matvar,"z",0,field);
    /* structure index 1 */
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,x2,0);
    Mat_VarSetStructFieldByName(matvar,"x",1,field);
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,y2,0);
    Mat_VarSetStructFieldByName(matvar,"y",1,field);
    field = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,&z2,MAT_F_COMPLEX);
    Mat_VarSetStructFieldByName(matvar,"z",1,field);
    Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return EXIT_SUCCESS;
}
