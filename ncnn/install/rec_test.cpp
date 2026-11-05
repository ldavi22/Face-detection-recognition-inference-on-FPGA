#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "net.h"
#include <cstdio>
#include <cmath>

int main()
{
    ncnn::Net net;
    if (net.load_param("mobilefacenet.param"))
    {
        printf("param failed\n");
        return 1;
    }
    if (net.load_model("mobilefacenet.bin"))
    {
        printf("binary failed \n");
        return 1;
    }

    const char* files[3] = {"1.jpg", "2.jpg", "2_1.jpg"};
    float emb[3][128];

    for (int i = 0; i < 3; i++)
    {
        int w, h, c;
        unsigned char* data = stbi_load(files[i], &w, &h, &c, 3);
        if (!data)
        {
            printf("%s: %s\n", files[i], stbi_failure_reason());
            return 1;
        }

        printf("%s: %dx%d c=%d\n", files[i], w, h, c);

        ncnn::Extractor ex = net.create_extractor();

        ncnn::Mat in = ncnn::Mat::from_pixels(data, ncnn::Mat::PIXEL_RGB2BGR, w, h);
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);

        float n = 0;
        for (int m = 0; m < 128; m++) n += out[m] * out[m];
        n = sqrtf(n);
        for (int k = 0; k < 128; k++) emb[i][k] = out[k] / n;

        stbi_image_free(data);
    }

    float same_people = 0;
    float diff_people = 0;
    for (int i = 0; i < 128; i++)
    {
        same_people += emb[1][i] * emb[2][i];
        diff_people += emb[0][i] * emb[1][i];
    }

    printf("Cosine similarity between: \n 1) Same people %.2f \n 2) different people %.2f", same_people, diff_people);
}
