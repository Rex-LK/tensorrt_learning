#ifndef GEMM_HPP
#define GEMM_HPP

#include <vector>
#include <initializer_list>
#include <ostream>
#include <vector>
/* 实现一个自定义的matrix类 */
class Matrix{
public:
    Matrix();
    Matrix(int rows, int cols, const std::initializer_list<float>& pdata={});
    Matrix(int rows, int cols, const std::vector<float>&v);
    
    const float& operator()(int irow, int icol)const {return data_[irow * cols_ + icol];}
    float& operator()(int irow, int icol){return data_[irow * cols_ + icol];}
    Matrix element_wise(const std::function<float(float)> &func) const;
    Matrix operator*(const Matrix &value) const;
    Matrix operator*(float value) const;

    Matrix operator+(float value) const;

    Matrix operator-(float value) const;
    Matrix operator/(float value) const;

    int rows() const{return rows_;}
    int cols() const{return cols_;}
    Matrix view(int rows, int cols) const;
    Matrix power(float y) const;
    float reduce_sum() const;
    float* ptr() const{return (float*)data_.data();}
    Matrix exp(float value);

    int rows_ = 0;
    int cols_ = 0;
    std::vector<float> data_;
};

/* 全局操作符重载，使得能够被cout << m; */
std::ostream& operator << (std::ostream& out, const Matrix& m);

#endif // GEMM_HPP