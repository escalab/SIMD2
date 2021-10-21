#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

/**
    count number of vertices, for host to allocate memory
**/
int count_from_pbbs(std::string fname){

    std::ifstream infile(fname);
    int vet_count = 0;
    std::string line;
    std::getline(infile, line);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a)) break;
        if (!(iss >> b)) break;
        vet_count = std::max(vet_count, std::max(a,b));
    }

    return vet_count;
}

/**
    conver  adj list liked data structure to ad mat
**/
int read_from_pbbs(std::string fname, int v, float * g){
    std::ifstream infile(fname);
    int vet_count = 0;
    std::string line;
    std::getline(infile, line);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int a, b;
        float w;
        if (!(iss >> a)) break;
        if (!(iss >> b)) break;
        if (!(iss >> w)) break;
        g[a*v+b] = w;
        g[b*v+a] = w;
    }

    //set all 0 to inf
    for(int i = 0; i < v; i++){
        for(int j = 0; j < v; j++){
            if(i != j){
                if(g[i*v+j] == 0) g[i*v+j] = FLT_MAX;
            }
        }
    }
    return vet_count;
}



