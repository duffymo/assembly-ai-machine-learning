def convolution_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1

if __name__ == "__main__":
    print(convolution_output_size(64, 13, 2, 3))