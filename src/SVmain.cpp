#include "SVApp.hpp"





int main(int argc, char* argv[])
{
	std::cout << "Started\n";

	SVAppConfig svcfg;

	SVApp svapp(svcfg);

	auto res = svapp.init();

	svapp.run();
	
	std::cout << "Done!\n";

	return 0;
}





