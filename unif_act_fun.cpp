#include <sycl/sycl.hpp>

using namespace sycl;

double unif_act_approx(
    double f_x, double g_x, 
    double ALPHA_1, double BETA_1, 
    double ALPHA_2, double BETA_2,
    size_t N){
    
    // A function that approximates Mittag-Leffler function
    // input: x
    // parameters: 
    //     ALPHA_1, BETA_1
    //     ALPHA_2, BETA_2
    //     N: controls the precision of the approximation;

    //initilize the vector for the numerator
    std::vector<double> num_vector(N, 1.0);
    
    //initilize the vector for the denominator
    std::vector<double> den_vector(N, 1.0);

     // create buffers accordingly
    buffer<double, 1> num_vector_buffer(num_vector.data(), range<1>(N));
    buffer<double, 1> den_vector_buffer(den_vector.data(), range<1>(N));

    //# Create a device queue with device selector
    queue q(gpu_selector_v);

    //# Print the device name
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    q.submit([&](handler &h) {

      //# Create accessor for vector_buffer
      accessor num_vector_accessor (num_vector_buffer, h);
      accessor den_vector_accessor (den_vector_buffer, h);

      h.parallel_for(range<1>(N), [=](id<1> k) {
         num_vector_accessor[k] =  sycl::pow(f_x, double(k)) / sycl::tgamma(ALPHA_1 * double(k) + BETA_1);
         den_vector_accessor[k] =  sycl::pow(g_x, double(k)) / sycl::tgamma(ALPHA_2 * double(k) + BETA_2);
      });
    }).wait();

    //# Create a host accessor to copy data from device to host
    {
        host_accessor h_a(num_vector_buffer, read_only);
        for(int i = 0; i < N; i++) std::cout << num_vector[i] << " ";
        std::cout << "" << std::endl;
        
        host_accessor h_b(den_vector_buffer, read_only);
        for(int i = 0; i < N; i++) std::cout << den_vector[i] << " ";
        std::cout << "" << std::endl;
    }

    std::cout << "now reducing.." << std::endl;

    // return the sum
    double sum = 0.0;
    return sum;
}

int main() {
    double a = unif_act_approx(1.0, 2.0, 3.0, 2.0, 1.0, 1.4, 10);
    std::cout << a << std::endl;

    return 0;
}