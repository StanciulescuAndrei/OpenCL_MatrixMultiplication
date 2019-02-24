kernel void MatrixMulti(global int * dimension, global double * a, global double * b, global double * c){
    int id = get_global_id(0);
    int NLin_1 = dimension[0]; //number of lines of first matrix
    int NCol_1 = dimension[1]; //number of columns of first matrix
    int NCol_2 = dimension[2]; //number of columns of second matrix

    int L = id / NCol_2; //get the position in the final matrix from the id
    int C = id - L*NCol_2;

    double element = 0;
    for(int i=0;i<NCol_1;i++){
        element = element + a[L*NCol_1 + i] * b[C + NCol_2*i];
    }
    c[id] = element;
    
}