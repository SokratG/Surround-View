#include <SVApp.hpp>



int main(int argc, char* argv[])
{

	SVAppConfig svcfg;

	SVApp svapp(svcfg);

	auto res = svapp.init();

	svapp.run();

	return 0;
}





