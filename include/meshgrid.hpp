/*
MIT License
Copyright (c) 2019 Xiaohong Chen
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MESHGEN_MESHGRID_HPP
#define MESHGEN_MESHGRID_HPP


#include <type_traits>
#include <iterator>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace meshgen {

#define CHECK_BAD_ARGUMENT(expression) \
    if (! (expression)) { \
        throw std::invalid_argument("bad argument"); \
    }

#define CHECK_BAD_SIZE(expression) \
    if (! (expression)) { \
        throw std::domain_error("bad size"); \
    }

    template<typename FwdIt,
        typename T,
        typename = std::enable_if_t<std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<FwdIt>::iterator_category>::value, int>
    >
        inline void linspace(FwdIt first, FwdIt last, const T& x1, const T& h)
    {
        CHECK_BAD_ARGUMENT(h >= T/*zero*/());
        T x = x1;
        for (; first != last; ++first)
        {
            *first = x;
            x += h;
        }
    }

    template<typename FwdIt,
        typename T,
        typename Size,
        typename = std::enable_if_t<std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<FwdIt>::iterator_category>::value>
    >
        inline void linspace(FwdIt first, const T& x1, const T& x2, Size N)
    {
        CHECK_BAD_ARGUMENT(N > Size/*zero*/());
        T x = x1;
        if ((N - 1) == Size/*zero*/()) {
            CHECK_BAD_ARGUMENT(x1 == x2);
            *first = x;
            return;
        }
        const T h = (x2 - x1) / (N - 1);
        for (Size i = 0; i < N; ++i)
        {
            *first = x;
            x += h;
            ++first;
        }
    }

    template<typename T, typename Size>
    inline std::vector<T> linspace(const T& x1, const T& x2, Size N) {
        typedef std::vector<T> vector_type;
        CHECK_BAD_ARGUMENT(N > Size/*zero*/());
        vector_type ret(N);
        if ((N - 1) == Size/*zero*/()) {
            CHECK_BAD_ARGUMENT(x1 == x2);
            ret[0] = x1;
        }
        else {
            const T h = (x2 - x1) / (N - 1);
            std::generate(ret.begin(), ret.end(),
                [x = x1 - h, h]() mutable {x += h; return x; });
        }
        return ret;
    }

    // D: which dimension, 0 for x, 1 for y, 3 for z
    // ND: number of dimensions
    template<class T, std::size_t D, std::size_t ND>
    class mesh_grid;

    template<class T, std::size_t D>
    class mesh_grid<T, D, 2> {
    public:
        typedef mesh_grid<T, D, 2> self_type;
        typedef std::size_t size_type;
        typedef T value_type;
        typedef const T& const_reference;
        typedef T& reference;
        typedef const T* const_pointer;
        typedef T* pointer;

        // Construction and destruction
        mesh_grid() {
            size1_ = 0;
            size2_ = 0;
            data_ = nullptr;
        }

        template <class IT>
        mesh_grid(size_type size1, size_type size2, IT first, IT last) {
            CHECK_BAD_ARGUMENT(last - first >= 0);
            check_size(size1, size2, static_cast<size_type>(last - first),
                std::integral_constant<size_type, D>());
            data_ = nullptr;
            if (last - first) {
                data_ = new value_type[last - first];
                std::copy(first, last, data_);
            }
            size1_ = size1;
            size2_ = size2;
        }

        mesh_grid(const self_type& m) {
            data_ = nullptr;
            if (m.size()) {
                data_ = new value_type[m.size()];
                std::copy(m.data_, m.data_ + m.size(), data_);
            }
            size1_ = m.size1_;
            size2_ = m.size2_;
        }

        mesh_grid(self_type&& m) {
            size1_ = 0;
            size2_ = 0;
            data_ = nullptr;
            swap(m);
        }

        ~mesh_grid() {
            if (size()) delete[] data_;
        }

        size_type size1() const {
            return size1_;
        }

        size_type size2() const {
            return size2_;
        }

        size_type size() const {
            return size_impl(std::integral_constant<size_type, D>());
        }

        const_pointer& data() const {
            return data_;
        }

        pointer& data() {
            return data_;
        }

        // Element access
        const_reference operator () (size_type i, size_type j) const {
            CHECK_BAD_SIZE(i < size1_);
            CHECK_BAD_SIZE(j < size2_);
            return at_element(i, j, std::integral_constant<size_type, D>());
        }

        // Assignment
        self_type& operator=(const self_type& m) {
            if (this != &m) {
                pointer p_data = nullptr;
                if (m.size()) {
                    p_data = new value_type[m.size()];
                    std::copy(m.data_, m.data_ + m.size(), p_data);
                }
                std::swap(p_data, data_);
                if (size()) {
                    delete[] p_data;
                }
                size1_ = m.size1_;
                size2_ = m.size2_;
            }
            return *this;
        }

        self_type& operator=(self_type&& m) {
            swap(m);
            return *this;
        }

        // Swap
        void swap(self_type& m) {
            if (this != &m) {
                std::swap(size1_, m.size1_);
                std::swap(size2_, m.size2_);
                std::swap(data_, m.data_);
            }
        }

        friend void swap(self_type& m1, self_type& m2) {
            m1.swap(m2);
        }

    private:
        void check_size(size_type size1, size_type, size_type size,
            std::integral_constant<size_type, 0>) {
            CHECK_BAD_ARGUMENT(size1 == size);
        }

        void check_size(size_type, size_type size2, size_type size,
            std::integral_constant<size_type, 1>) {
            CHECK_BAD_ARGUMENT(size2 == size);
        }

        const_reference at_element(size_type i, size_type,
            std::integral_constant<size_type, 0>) const {
            return data_[i];
        }

        const_reference at_element(size_type, size_type j,
            std::integral_constant<size_type, 1>) const {
            return data_[j];
        }

        size_type size_impl(std::integral_constant<size_type, 0>) const {
            return size1_;
        }

        size_type size_impl(std::integral_constant<size_type, 1>) const {
            return size2_;
        }

        size_type size1_;
        size_type size2_;
        pointer data_;
    };

    template<class T, std::size_t D>
    class mesh_grid<T, D, 3> {
    public:
        typedef mesh_grid<T, D, 3> self_type;
        typedef std::size_t size_type;
        typedef T value_type;
        typedef const T& const_reference;
        typedef T& reference;
        typedef const T* const_pointer;
        typedef T* pointer;

        // Construction and destruction
        mesh_grid() {
            size1_ = 0;
            size2_ = 0;
            size3_ = 0;
            data_ = nullptr;
        }

        template<class IT>
        mesh_grid(size_type size1, size_type size2, size_type size3, IT first, IT last) {
            CHECK_BAD_ARGUMENT(last - first >= 0);
            check_size(size1, size2, size3, static_cast<size_type>(last - first),
                std::integral_constant<size_type, D>());
            if (last - first) {
                data_ = new value_type[last - first];
                std::copy(first, last, data_);
            }
            size1_ = size1;
            size2_ = size2;
            size3_ = size3;
        }

        mesh_grid(const self_type& m) {
            if (m.size()) {
                data_ = new value_type[m.size()];
                std::copy(m.data_, m.data_ + m.size(), data_);
            }
            size1_ = m.size1_;
            size2_ = m.size2_;
            size3_ = m.size3_;
        }

        mesh_grid(self_type&& m) {
            size1_ = 0;
            size2_ = 0;
            size3_ = 0;
            data_ = nullptr;
            swap(m);
        }

        ~mesh_grid() {
            if (size()) delete[] data_;
        }

        size_type size1() const {
            return size1_;
        }

        size_type size2() const {
            return size2_;
        }

        size_type size3() const {
            return size3_;
        }

        size_type size() const {
            return size_impl(std::integral_constant<size_type, D>());
        }

        const_pointer& data() const {
            return data_;
        }

        pointer& data() {
            return data_;
        }

        // Element access
        const_reference operator () (size_type i, size_type j, size_type k) const {
            CHECK_BAD_SIZE(i < size1_);
            CHECK_BAD_SIZE(j < size2_);
            CHECK_BAD_SIZE(k < size3_);
            return at_element(i, j, k, std::integral_constant<size_type, D>());
        }

        // Assignment
        self_type& operator=(const self_type& m) {
            if (this != &m) {
                pointer p_data = nullptr;
                if (m.size()) {
                    p_data = new value_type[m.size()];
                    std::copy(m.data_, m.data_ + m.size(), p_data);
                }
                std::swap(p_data, data_);
                if (size()) {
                    delete[] p_data;
                }
                size1_ = m.size1_;
                size2_ = m.size2_;
                size3_ = m.size3_;
            }
            return *this;
        }

        self_type& operator=(self_type&& m) {
            swap(m);
            return *this;
        }

        // Swap
        void swap(self_type& m) {
            if (this != &m) {
                std::swap(size1_, m.size1_);
                std::swap(size2_, m.size2_);
                std::swap(size3_, m.size3_);
                std::swap(data_, m.data_);
            }
        }

        friend void swap(self_type& m1, self_type& m2) {
            m1.swap(m2);
        }

    private:
        void check_size(size_type size1, size_type, size_type, size_type size,
            std::integral_constant<size_type, 0>) {
            CHECK_BAD_ARGUMENT(size1 == size);
        }

        void check_size(size_type, size_type size2, size_type, size_type size,
            std::integral_constant<size_type, 1>) {
            CHECK_BAD_ARGUMENT(size2 == size);
        }

        void check_size(size_type, size_type, size_type size3, size_type size,
            std::integral_constant<size_type, 2>) {
            CHECK_BAD_ARGUMENT(size3 == size);
        }

        const_reference at_element(size_type i, size_type, size_type,
            std::integral_constant<size_type, 0>) const {
            return data_[i];
        }

        const_reference at_element(size_type, size_type j, size_type,
            std::integral_constant<size_type, 1>) const {
            return data_[j];
        }

        const_reference at_element(size_type, size_type, size_type k,
            std::integral_constant<size_type, 2>) const {
            return data_[k];
        }

        size_type size_impl(std::integral_constant<size_type, 0>) const {
            return size1_;
        }

        size_type size_impl(std::integral_constant<size_type, 1>) const {
            return size2_;
        }

        size_type size_impl(std::integral_constant<size_type, 2>) const {
            return size3_;
        }

        size_type size1_;
        size_type size2_;
        size_type size3_;
        pointer data_;
    };

    template <class V>
    inline auto meshgrid(const V& x, const V& y) {
        typedef typename V::value_type value_type;
        typedef typename V::size_type size_type;
        typedef mesh_grid<value_type, 0, 2> x_grid_type;
        typedef mesh_grid<value_type, 1, 2> y_grid_type;
        typedef std::tuple<x_grid_type, y_grid_type> return_type;

        size_type M = x.size();
        size_type N = y.size();
        return return_type(x_grid_type(M, N, x.begin(), x.end()),
            y_grid_type(M, N, y.begin(), y.end()));
    }

    template <class IT>
    inline auto meshgrid(IT x_first, IT x_last, IT y_first, IT y_last) {
        typedef typename std::iterator_traits<IT>::value_type value_type;
        typedef mesh_grid<value_type, 0, 2> x_grid_type;
        typedef mesh_grid<value_type, 1, 2> y_grid_type;
        typedef typename x_grid_type::size_type size_type;
        typedef std::tuple<x_grid_type, y_grid_type> return_type;

        CHECK_BAD_ARGUMENT(x_last - x_first >= 0);
        CHECK_BAD_ARGUMENT(y_last - y_first >= 0);
        size_type M = static_cast<size_type>(x_last - x_first);
        size_type N = static_cast<size_type>(y_last - y_first);
        return return_type(x_grid_type(M, N, x_first, x_last),
            y_grid_type(M, N, y_first, y_last));
    }

    template <class V>
    inline auto meshgrid(const V& x,
        const V& y,
        const V& z) {
        typedef typename V::value_type value_type;
        typedef typename V::size_type size_type;
        typedef mesh_grid<value_type, 0, 3> x_grid_type;
        typedef mesh_grid<value_type, 1, 3> y_grid_type;
        typedef mesh_grid<value_type, 2, 3> z_grid_type;
        typedef std::tuple<x_grid_type, y_grid_type, z_grid_type> return_type;

        size_type L = x.size();
        size_type M = y.size();
        size_type N = z.size();

        return return_type(x_grid_type(L, M, N, x.begin(), x.end()),
            y_grid_type(L, M, N, y.begin(), y.end()),
            z_grid_type(L, M, N, z.begin(), z.end()));
    }

    template <class IT>
    inline auto meshgrid(IT x_first, IT x_last,
        IT y_first, IT y_last,
        IT z_first, IT z_last) {
        typedef typename std::iterator_traits<IT>::value_type value_type;
        typedef mesh_grid<value_type, 0, 3> x_grid_type;
        typedef mesh_grid<value_type, 1, 3> y_grid_type;
        typedef mesh_grid<value_type, 2, 3> z_grid_type;
        typedef typename x_grid_type::size_type size_type;
        typedef std::tuple<x_grid_type, y_grid_type, z_grid_type> return_type;

        CHECK_BAD_ARGUMENT(x_last - x_first >= 0);
        CHECK_BAD_ARGUMENT(y_last - y_first >= 0);
        CHECK_BAD_ARGUMENT(z_last - z_first >= 0);
        size_type L = static_cast<size_type>(x_last - x_first);
        size_type M = static_cast<size_type>(y_last - y_first);
        size_type N = static_cast<size_type>(z_last - z_first);

        return return_type(x_grid_type(L, M, N, x_first, x_last),
            y_grid_type(L, M, N, y_first, y_last),
            z_grid_type(L, M, N, z_first, z_last));
    }


} // meshgen

#endif
