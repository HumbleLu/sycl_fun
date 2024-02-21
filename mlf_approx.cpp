#include <sycl/sycl.hpp>

using namespace sycl;

double mlf_approx(double z, double ALPHA, double BETA, size_t N){
    // A function that approximates Mittag-Leffler function
    // input: z
    // parameters: 
    //     ALPHA;
    //     BETA;
    //     N: controls the precision of the approximation;

    //initilize the vector
    std::vector<double> vector(N, 1.0);

     // create a buffer
    buffer<double, 1> vector_buffer(vector.data(), range<1>(N));

    //# Create a device queue with device selector
    queue q(gpu_selector_v);

    //# Print the device name
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Initilize sum
    double sum = 0.0;
    {
        // Create a buffer for sum to get the reduction results
        buffer<double> sum_buf{&sum, 1};

        // Submit a SYCL kernel into a queue
        q.submit([&](handler &cgh) {
          // We can use built-in reduction primitive
          auto sum_reduction = reduction(sum_buf, cgh, plus<double>());

          //# Create accessor for vector_buffer
          accessor vector_accessor (vector_buffer, cgh);

          // A reference to the reducer is passed to the lambda
          cgh.parallel_for(range<1>{N}, sum_reduction,
                          [=](id<1> k, auto &reducer) {
                          reducer.combine(
                            sycl::pow(z, double(k)) / sycl::tgamma(ALPHA * double(k) + BETA)
                          ); });
        }).wait();
    }

    // return the sum
    return sum;
}

int main() {
    // test
    std::cout << "first" << std::endl;
    double a = mlf_approx(1.0, 2.0, 3.0, 2);
    std::cout << a << std::endl;
    
    std::cout << "seconde" << std::endl;
    double b = mlf_approx(1.0, 2.0, 3.0, 100);
    std::cout << b << std::endl;

    return 0;
}