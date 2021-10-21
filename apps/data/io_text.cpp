#include "from_pbbs.h"

int main(){
    int vertex_count = count_from_pbbs("32_inputs_pbbs.txt") + 1;
    std::cout << "number of vertices: " << vertex_count << "\n";

    float * g  = new float[vertex_count * vertex_count];

    read_from_pbbs("32_inputs_pbbs.txt",vertex_count,g );

    for(int i = 0; i < vertex_count; i++){
        for(int j = 0; j < vertex_count; j++){
            if (g[i*vertex_count+j] < FLT_MAX -100)
                std::cout << g[i*vertex_count+j] << " ";
            else std::cout << "inf ";
        }
        std::cout << std::endl;
    }

    delete [] g;

    
}