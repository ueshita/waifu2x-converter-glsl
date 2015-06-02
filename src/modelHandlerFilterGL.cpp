
#include <fstream>
#include <sstream>
#include "modelHandler.hpp"
#include "filterGL.h"

static std::string loadFile(const char *path);

static bool loadShader(const char *preDefine, const char *vsPath, const char *fsPath, GLuint *prog);

bool w2xc::Model::loadGLShader()
{
	std::ostringstream preDefine;
	preDefine << "#version 140\n";
	preDefine << "#define NUM_INPUT_PLANES	" << getNInputPlanes() << std::endl;
	preDefine << "#define NUM_OUTPUT_PLANES	" << getNOutputPlanes() << std::endl;

	if (!loadShader(preDefine.str().c_str(), "shaders/waifu2x_vs.glsl", "shaders/waifu2x_fs.glsl", &shader.program)) {
		std::cout << "GL shader compile error." << std::endl;
		return false;
	}
	shader.a_position = glGetAttribLocation(shader.program, "a_position");
	shader.a_texCoord = glGetAttribLocation(shader.program, "a_texCoord");
	shader.bias          = glGetUniformLocation(shader.program, "bias");
	shader.weightMatrix  = glGetUniformLocation(shader.program, "weightMatrix");
	shader.inputTextures = glGetUniformLocation(shader.program, "inputTextures");

	return true;
}

bool w2xc::Model::filterGL(int modelIndex)
{
	// filter core process
	return filterGLProcess(shader, nInputPlanes, nOutputPlanes, weights, biases, modelIndex);
}


static bool loadFile(const char *path, std::string& outstr)
{
	std::ifstream file;
	file.open(path);
	if (!file.is_open()) {
		return false;
	}
	std::istreambuf_iterator<char> it(file);
	std::istreambuf_iterator<char> last;
	outstr = std::string(it, last);
	return true;
}

static bool loadShader(const char *preCode, const char *vsPath, const char *fsPath, GLuint *prog)
{
	std::string vsCode, fsCode;
	
	// Loading shader code
	if (!loadFile(vsPath, vsCode)) return false;
	if (!loadFile(fsPath, fsCode)) return false;

	// Create shaders
	GLuint vsh = glCreateShader(GL_VERTEX_SHADER);
	GLuint fsh = glCreateShader(GL_FRAGMENT_SHADER);

	const char *vsCodePtr[2] = {preCode, vsCode.c_str()};
	const char *fsCodePtr[2] = {preCode, fsCode.c_str()};
	
	GLint linked = 0, compiled = 0;
	
	// Compiling vertex shader
	glShaderSource(vsh, 2, vsCodePtr, 0);
	glCompileShader(vsh);
	glGetShaderiv(vsh, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE) {
		char log[512];
		glGetShaderInfoLog(vsh, sizeof(log), 0, log);
		std::cout << log << std::endl;
		glDeleteShader(vsh);
		glDeleteShader(fsh);
	    return false;
	}

	// Compiling fragment shader
	glShaderSource(fsh, 2, fsCodePtr, 0);
	glCompileShader(fsh);
	glGetShaderiv(fsh, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE) {
		char log[512];
		glGetShaderInfoLog(fsh, sizeof(log), 0, log);
		std::cout << log << std::endl;
		glDeleteShader(vsh);
		glDeleteShader(fsh);
	    return false;
	}

	*prog = glCreateProgram();
	glAttachShader(*prog, vsh);
	glAttachShader(*prog, fsh);
	glLinkProgram(*prog);

	glDeleteShader(vsh);
	glDeleteShader(fsh);
	
	glGetProgramiv(*prog, GL_LINK_STATUS, &linked);
	if (linked == GL_FALSE) {
		glDeleteProgram(*prog);
		return false;
	}

	return true;
}
