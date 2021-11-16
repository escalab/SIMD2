#include <float.h>
template<typename T>
T check_sum(T* data, int n){
    // check sum:
    T checksum = 0;
    int valid_count = 0;
    for (int i = 0; i < n; i++){
        if (abs(data[i]) < (FLT_MAX-100)){
            checksum += abs(data[i]);
            valid_count += 1;
        }
    }
    std::cout << valid_count << " entry valid\n";
    checksum = sqrt(checksum);
    return checksum;
}