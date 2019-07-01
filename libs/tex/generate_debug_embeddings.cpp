/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <math/vector.h>

#include "debug.h"

TEX_NAMESPACE_BEGIN

const bool font[] = {
    0,1,0, 0,1,0, 1,1,0, 1,1,0, 1,0,0, 1,1,1, 0,1,0, 1,1,1, 0,1,0 ,0,1,0,
    1,0,1, 1,1,0, 0,0,1, 0,0,1, 1,0,1, 1,0,0, 1,0,0, 0,0,1, 1,0,1, 1,0,1,
    1,0,1, 0,1,0, 0,1,0, 0,1,0, 1,1,1, 1,1,0, 1,1,0, 0,0,1, 0,1,0, 0,1,1,
    1,0,1, 0,1,0, 1,0,0, 0,0,1, 0,0,1, 0,0,1, 1,0,1, 0,1,0, 1,0,1, 0,0,1,
    0,1,0, 1,1,1, 1,1,1, 1,1,0, 0,0,1, 1,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0
};

void print_number(mve::ByteImage::Ptr image, int x, int y, int digit, math::Vec3uc color) {
    assert(0 <= x && x < image->width() - 3);
    assert(0 <= y && y < image->height() - 5);
    assert(0 <= digit && digit <= 9);

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 5; ++j) {
            if(font[30 * j + digit * 3 + i]) {
                for(int c = 0; c < image->channels(); ++c) {
                    image->at(x+i, y+j, c) = color[c];
                }
            }
        }
    }
}

void
generate_debug_colors(std::vector<math::Vec4f> & colors) {
    for (float s = 1.0f; s > 0.0f; s -= 0.4) {
        for (float v = 1.0f; v > 0.0f; v -= 0.3) {
            for (float h = 0.0f; h < 360.0f; h += 30.0f) {
                float c = v * s;
                float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
                float m = v - c;

                math::Vec4f color;
                if (0 <= h && h < 60)
                    color = math::Vec4f(c, x, 0.0f, 1.0f);
                if (60 <= h && h < 120)
                    color = math::Vec4f(x, c, 0.0f, 1.0f);
                if (120 <= h && h < 180)
                    color = math::Vec4f(0.0f, c, x, 1.0f);
                if (180 <= h && h < 240)
                    color = math::Vec4f(0.0f, x, c, 1.0f);
                if (240 <= h && h < 300)
                    color = math::Vec4f(x, 0.0f, c, 1.0f);
                if (300 <= h && h < 360)
                    color = math::Vec4f(c, 0.0f, x, 1.0f);

                color = color + math::Vec4f(m, m, m, 0.0f);
                colors.push_back(color);
            }
        }
    }
}

void
generate_debug_embeddings(std::vector<TextureView> * texture_views) {
    std::vector<math::Vec4f> colors;
    generate_debug_colors(colors);

    #pragma omp parallel for
    for (std::size_t i = 0; i < texture_views->size(); ++i) {
        math::Vec4f float_color =  colors[i % colors.size()];

        TextureView * texture_view = &(texture_views->at(i));

        /* Determine font color depending on luminance of background. */
        float luminance = math::interpolate(float_color[0], float_color[1], float_color[2], 0.30f, 0.59f, 0.11f);
        math::Vec3uc font_color = luminance > 0.5f ? math::Vec3uc(0,0,0) : math::Vec3uc(255,255,255);

        math::Vec3uc color;
        color[0] = float_color[0] * 255.0f;
        color[1] = float_color[1] * 255.0f;
        color[2] = float_color[2] * 255.0f;

        mve::ByteImage::Ptr image = mve::ByteImage::create(texture_view->get_width(), texture_view->get_height(), 3);
        image->fill_color(*color);

        for(int ox=0; ox < image->width() - 13; ox += 13) {
            for(int oy=0; oy < image->height() - 6; oy += 6) {
                std::size_t id = texture_view->get_id();
                int d0 = id / 100;
                int d1 = (id % 100) / 10;
                int d2 = id % 10;

                print_number(image, ox, oy, d0, font_color);
                print_number(image, ox + 4, oy, d1, font_color);
                print_number(image, ox + 8, oy, d2, font_color);
            }
        }

        texture_view->bind_image(image);
    }
}

void
generate_segmentation_embeddings(std::vector<TextureView> * texture_views) {

    // colors taken from segmentation program. They are in BGR order!
    // atts = [0 'bg', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
    //         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

    std::vector<math::Vec3uc> seg_colors{
        {255, 0, 0}, {255, 85, 0}, {255, 170, 0},
        {255, 0, 85}, {255, 0, 170},
        {0, 255, 0}, {85, 255, 0}, {170, 255, 0},
        {0, 255, 85}, {0, 255, 170},
        {0, 0, 255}, {85, 0, 255}, {170, 0, 255},
        {0, 85, 255}, {0, 170, 255},
        {255, 255, 0}, {255, 255, 85}, {255, 255, 170},
        {255, 0, 255}, {255, 85, 255}, {255, 170, 255},
        {0, 255, 255}, {85, 255, 255}, {170, 255, 255}};

    #pragma omp parallel for
    for (std::size_t i = 0; i < texture_views->size(); ++i) {
        TextureView * texture_view = &(texture_views->at(i));

        mve::ByteImage::Ptr seg_image = texture_view->get_segmentation_image();
        mve::ByteImage::Ptr image = mve::ByteImage::create(texture_view->get_width(), texture_view->get_height(), 3);
        for(std::size_t idx = 0; idx < image->width() * image->height() * 3; idx += 3) {
                math::Vec3uc & color = seg_colors[seg_image->at(idx / 3, 0) % seg_colors.size()];
                image->at(idx) = color[2];
                image->at(idx+1) = color[1];
                image->at(idx+2) = color[0];
        }

        math::Vec3uc font_color = math::Vec3uc(0,0,0);

        for(int ox=0; ox < image->width() - 13; ox += 13) {
            for(int oy=0; oy < image->height() - 6; oy += 6) {
                std::size_t id = texture_view->get_id();
                int d0 = id / 100;
                int d1 = (id % 100) / 10;
                int d2 = id % 10;

                print_number(image, ox, oy, d0, font_color);
                print_number(image, ox + 4, oy, d1, font_color);
                print_number(image, ox + 8, oy, d2, font_color);
            }
        }

        texture_view->bind_image(image);

        // texture_views->at(i).bind_segmentation_image();
    }
}


TEX_NAMESPACE_END
