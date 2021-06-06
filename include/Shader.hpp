#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

#include <GLES3/gl32.h>
#include <EGL/egl.h>

#include <glm/glm.hpp>



class Shader
{
public:
        Shader() = default;

	~Shader() {
            if (isInit) {
                shadersRelease();
                programRelease();
            }
	}
	bool initShader(const char* vertexfile, const char* fragmentfile, const char* geometryfile = nullptr) {
            if (vertexfile == nullptr || fragmentfile == nullptr)
                    return false;

            vertexshader = initShaders(GL_VERTEX_SHADER, vertexfile);

            fragmentshader = initShaders(GL_FRAGMENT_SHADER, fragmentfile);

            if (vertexshader == 0 || fragmentshader == 0)
                return false;

            if (geometryfile != nullptr)
                    geometryshader = initShaders(GL_GEOMETRY_SHADER, geometryfile);


            shaderprogram = initProgram();
            if (shaderprogram == 0)
                return false;


            isInit = true;

            return isInit;
	}

	bool useProgramm() {
            if (isInit == false)
                    return false;
            if (shaderprogram == 0)
                    return false;
            glUseProgram(shaderprogram);
            return true;
	}
	// ------------------------------------------------------------------------
	void setBool(const std::string& name, bool value) const
	{
            if (isInit)
                glUniform1i(glGetUniformLocation(shaderprogram, name.c_str()), (int)value);
	}
	// ------------------------------------------------------------------------
	void setInt(const std::string& name, int value) const
	{
            if (isInit)
                glUniform1i(glGetUniformLocation(shaderprogram, name.c_str()), value);
	}
	// ------------------------------------------------------------------------
	void setFloat(const std::string& name, float value) const
	{
            if (isInit)
                glUniform1f(glGetUniformLocation(shaderprogram, name.c_str()), value);
	}

	// ------------------------------------------------------------------------
	void setVec2(const std::string& name, const glm::vec2& value) const
	{
            glUniform2fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, &value[0]);
	}
	// ------------------------------------------------------------------------
	void setVec2(const std::string& name, float x, float y) const
	{
            glUniform2f(glGetUniformLocation(shaderprogram, name.c_str()), x, y);
	}
	// ------------------------------------------------------------------------
	void setVec3(const std::string& name, const glm::vec3& value) const
	{
            glUniform3fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, &value[0]);
	}
	void setVec3(const std::string& name, float x, float y, float z) const
	{
            glUniform3f(glGetUniformLocation(shaderprogram, name.c_str()), x, y, z);
	}
	// ------------------------------------------------------------------------
	void setVec4(const std::string& name, const glm::vec4& value) const
	{
            glUniform4fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, &value[0]);
	}
	void setVec4(const std::string& name, float x, float y, float z, float w)
	{
            glUniform4f(glGetUniformLocation(shaderprogram, name.c_str()), x, y, z, w);
	}
	// ------------------------------------------------------------------------
	void setMat2(const std::string& name, const glm::mat2& mat) const
	{
            glUniformMatrix2fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}
	// ------------------------------------------------------------------------
	void setMat3(const std::string& name, const glm::mat3& mat) const
	{
            glUniformMatrix3fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}
	// ------------------------------------------------------------------------
	void setMat4(const std::string& name, const glm::mat4& mat) const
	{
            glUniformMatrix4fv(glGetUniformLocation(shaderprogram, name.c_str()), 1, GL_FALSE, &mat[0][0]); // &mat[0][0] <-> glm::value_ptr(mat)
	}

	GLuint getShaderProgram() const { return shaderprogram; }
	GLuint getVertexShader() const { return vertexshader; }
	GLuint getFragmentShader() const { return fragmentshader; }
protected:
	GLuint initProgram() {
            GLuint program = glCreateProgram();
            glAttachShader(program, vertexshader); glAttachShader(program, fragmentshader);
            if (geometryshader != 0)
                glAttachShader(program, geometryshader);
            glLinkProgram(program);
            GLint linked;
            glGetProgramiv(program, GL_LINK_STATUS, &linked);
            if (linked) glUseProgram(program);
            else {
                programErrors(program);
                exit(EXIT_FAILURE);
            }
            return program;
	}
	GLuint initShaders(GLenum type, const char* filename) {
            GLuint shader = glCreateShader(type);
            GLint compiled;
            std::string str = Shader::readShaderSource(filename);
            const char* cstr = str.c_str();
            glShaderSource(shader, 1, &cstr, NULL);
            glCompileShader(shader);     
            glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
            if (!compiled) {
                shaderErrors(shader);
                exit(EXIT_FAILURE);
            }
            return shader;
	}
	std::string readShaderSource(const char* filename) {
            std::string str, ret = ""; std::ifstream in;
            in.open(filename);
            // std::stringstream str_; str_ << in.rdbuf(); // ok
            if (in.is_open()) {
                getline(in, str);
                while (in) {
                    ret += str + "\n";
                    getline(in, str);
                }
                return ret;
            }
            else {
                std::cerr << "Unable to Open File " << filename << "\n";
                exit(EXIT_FAILURE);
            }
	}
	void releaseShader(GLuint shader)
	{
            GLint deleted;
            glDeleteShader(shader);
            glGetShaderiv(shader, GL_DELETE_STATUS, &deleted);
            if (!deleted) {
                shaderDelErrors(shader);
                /*exit(EXIT_FAILURE)*/;
            }
	}
	

private:
	void programErrors(const GLint program) {
            GLint length;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
            GLchar log[buff_len]{ 0 };
            glGetProgramInfoLog(program, length, &length, log);
            std::cerr << "Shader Programm Error, Log Below\n" << log << "\n";
	}
	void shaderErrors(const GLint shader) {
            GLint length;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
            GLchar log[buff_len]{ 0 };
            glGetShaderInfoLog(shader, length, &length, log);
            std::cerr << "Compile Error, Log Below\n" << log << "\n";
	}
	void shaderDelErrors(const GLint shader) {
            GLint length;
            GLchar log[buff_len]{ 0 };
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
            glGetShaderInfoLog(shader, length, &length, log);
            std::cerr << "Delete Shader Error, Log Below\n" << log << "\n";
	}
	void shadersRelease() {
            releaseShader(vertexshader);
            releaseShader(fragmentshader);
	}
	void programRelease()
	{
            GLint deleted;
            glDeleteProgram(shaderprogram);
            glGetShaderiv(shaderprogram, GL_DELETE_STATUS, &deleted);
            if (!deleted) {
                std::cerr << "Error delete shader program # " << shaderprogram << "\n";
                /*exit(EXIT_FAILURE)*/;
            }
	}

private:
	GLuint shaderprogram = 0;
	GLuint vertexshader = 0;
	GLuint fragmentshader = 0;
	GLuint geometryshader = 0;
	GLboolean isInit = false;
	static constexpr size_t buff_len = 512;
};
