#include "MyForm.h"

using namespace System;
using namespace System::Windows::Forms;

typedef int(__cdecl* PInit)();
typedef void(__cdecl* PStartWorkers)();

[STAThread]
int main() {
	HMODULE h = LoadLibraryW(L"TEST_FUNC.dll");
	auto AgpInit = (PInit)GetProcAddress(h, "AgpInit");
	auto AgpWaitStartAndRun = (PStartWorkers)GetProcAddress(h, "AgpWaitStartAndRun");

	const int rank = AgpInit();

	if (!rank) {
		Application::EnableVisualStyles();
		Application::SetCompatibleTextRenderingDefault(false);
		Application::Run(gcnew TESTAGP::MyForm(h));
	}
	else {
		AgpWaitStartAndRun();
	}

	return 0;
}
