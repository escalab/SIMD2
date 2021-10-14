#include <float.h>
template<typename T>
T check_sum(T* data, int n){
        // check sum:
        T checksum = 0;
        for (int i = 0; i < n; i++){
                if (data[i] <= FLT_MAX && data[i] >= FLT_MAX - 5.0){
                    continue;
                }
                checksum += data[i];
            
        }
        checksum = sqrt(checksum);
        return checksum;
}