#include "kernel.h"
int main()
{
    const float pi=3.1415927;
    const int n=150;
    const float h=2*pi/n;
    std::tuple<std::vector<float>,std::vector<float>>  arrays=init(h,pi,n);
    run(std::get<0>(arrays).data(),std::get<1>(arrays).data(),
        n,h);
    for (const auto &i :std::get<1>(arrays))
    {
        std::cout << i<<" ";
    }

    return 0;
}