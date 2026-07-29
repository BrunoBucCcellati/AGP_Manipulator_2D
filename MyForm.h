#pragma onceFrame target

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#include <kdl/joint.hpp>
#include <kdl/segment.hpp>

#include <trac_ik/trac_ik.hpp>

using namespace System;
using namespace System::Drawing;
using namespace System::Windows::Forms;
using namespace System::Collections::Generic;
using namespace System::Drawing::Drawing2D;
using namespace System::Runtime::CompilerServices;

typedef void(__cdecl* P_MANIP)(
	int, bool, float, float, float, int, float, bool, float,
	unsigned int, float, float,
	const float*, int,
	float**, float*, float*, float*, size_t*, float*,
	int, const float*);

typedef void(__cdecl* P_FREE)(float*);

typedef void(__cdecl* P_START)(
	int, bool, float, float, float, int, float, bool, float,
	unsigned int, float, float,
	const float*, int,
	int,
	const float*,
	const float*);

typedef void(__cdecl* P_BUILD_TRAJECTORY)(
	int, bool, float,
	const float*,
	const float*,
	int, float, bool, float, unsigned int,
	float, float,
	const float*, int,
	float**, int*, size_t*);

namespace TESTAGP
{
	public enum class DemoMode
	{
		Positioning = 0,
		TrajectoryPlanning = 1
	};

	public enum class DragHandle
	{
		None = 0,
		Target = 1,
		Start = 2
	};

	public ref class PoseSnapshot sealed
	{
	public:
		PoseSnapshot(int n)
		{
			Angles = gcnew array<float>(n);
			Lengths = gcnew array<float>(n);
			EndX = 0.0f;
			EndY = 0.0f;
			BestF = 0.0f;
			Iterations = 0;
			AchievedEps = 0.0f;
			Micros = 0.0f;
		}

		array<float>^ Angles;
		array<float>^ Lengths;
		float EndX;
		float EndY;
		float BestF;
		int Iterations;
		float AchievedEps;
		float Micros;
	};

	public ref class MyForm sealed : public Form
	{
	public:
		MyForm(HMODULE hLib) : hLib(hLib)
		{
			this->SetStyle(ControlStyles::AllPaintingInWmPaint |
				ControlStyles::UserPaint |
				ControlStyles::OptimizedDoubleBuffer, true);
			this->Text = L"AGP Manipulator 2D";
			this->ClientSize = System::Drawing::Size(1200, 800);
			this->Resize += gcnew EventHandler(this, &MyForm::OnResize);
			this->MouseDown += gcnew MouseEventHandler(this, &MyForm::OnMouseDownPoint);
			this->MouseMove += gcnew MouseEventHandler(this, &MyForm::OnMouseMovePoint);
			this->MouseUp += gcnew MouseEventHandler(this, &MyForm::OnMouseUpPoint);

			fManip = reinterpret_cast<P_MANIP>(GetProcAddress(hLib, "AGP_Manip2D"));
			pFree = reinterpret_cast<P_FREE>(GetProcAddress(hLib, "AGP_Free"));
			pStart = reinterpret_cast<P_START>(GetProcAddress(hLib, "AgpStartManipND"));
			pBuildTrajectory = reinterpret_cast<P_BUILD_TRAJECTORY>(GetProcAddress(hLib, "AGP_BuildTransitionTrajectory"));

			angles = gcnew List<float>(8);
			lengths = gcnew List<float>(8);
			obstacleX = gcnew List<float>(4);
			obstacleY = gcnew List<float>(4);
			obstacleHalf = gcnew List<float>(4);
			plannedPoses = gcnew List<PoseSnapshot^>();
			animationFrames = gcnew List<PoseSnapshot^>();
			trajectoryPathWorld = gcnew List<PointF>();

			demoMode = DemoMode::Positioning;

			InitGraphicsResources();
			InitUI();
			InitAnimation();
			ResetRandomConfig();
			SetDefaultStartPoint(true);
			UpdateTrajectoryUiState();
		}

	private:
		literal float ObstacleClearance = 0.05f;
		literal int AnimationFramesPerSegment = 10;
		literal int AnimationIntervalMs = 16;
		literal float TransitionLengthEnergyWeight = 0.35f;
		literal float TransitionPrefixEnergyWeight = 0.175f;
		literal float PI_2 = 1.57079632679489661923f;

		ComboBox^ cbDemoMode;
		ComboBox^ cbBackend;
		CheckBox^ cbVarLen;
		CheckBox^ cbAdaptive;
		NumericUpDown^ nudMaxTheta;
		NumericUpDown^ nudBaseLength;
		NumericUpDown^ nudStretchFactor;
		NumericUpDown^ nudTargetX;
		NumericUpDown^ nudTargetY;
		NumericUpDown^ nudStartX;
		NumericUpDown^ nudStartY;
		NumericUpDown^ nudMaxIter;
		NumericUpDown^ nudR;
		TextBox^ txtEps;
		float currentEps;
		Button^ btnAdd;
		Button^ btnRem;
		Button^ btnOptimize;
		Button^ btnGenerateObstacles;
		Button^ btnClearObstacles;
		Label^ lblInfo;
		Label^ lblStartX;
		Label^ lblStartY;

		P_BUILD_TRAJECTORY pBuildTrajectory;
		initonly HMODULE hLib;
		initonly P_MANIP fManip;
		initonly P_FREE pFree;
		initonly P_START pStart;

		Pen^ obstaclePen;
		Pen^ obstacleMarginPen;
		SolidBrush^ obstacleBrush;
		Pen^ wallPen;
		Pen^ dashedPen;
		Pen^ targetPen;
		Pen^ startPen;
		Pen^ pathPen;
		Pen^ penRod;
		SolidBrush^ jointBrush;
		SolidBrush^ waypointBrush;
		HatchBrush^ wallHatchBrush;
		System::Drawing::Font^ uiFontBold11;
		System::Drawing::Font^ uiFontTextBox;
		System::Drawing::Font^ uiFontBold10;

		Timer^ animationTimer;

		DemoMode demoMode;
		int nSegments;
		bool variableLengths;
		List<float>^ angles;
		List<float>^ lengths;
		List<float>^ obstacleX;
		List<float>^ obstacleY;
		List<float>^ obstacleHalf;
		List<PoseSnapshot^>^ plannedPoses;
		List<PoseSnapshot^>^ animationFrames;
		List<PointF>^ trajectoryPathWorld;

		UInt32 rngState = 0xA5C39E0Du;
		DragHandle activeDragHandle = DragHandle::None;
		bool updatingFromMouse = false;
		bool syncingDefaultStartPoint = false;
		bool startPointCustomized = false;
		int animationFrameIndex = 0;
		bool animationRunning = false;

		static float WrapPi(float a)
		{
			const float PI = 3.14159265358979323846f;
			const float TWO_PI = 6.28318530717958647692f;
			while (a > PI) a -= TWO_PI;
			while (a < -PI) a += TWO_PI;
			return a;
		}

		static float WrappedDelta(float from, float to)
		{
			return WrapPi(to - from);
		}

		static float LerpWrappedAngle(float from, float to, float t)
		{
			return WrapPi(from + t * WrappedDelta(from, to));
		}

		static float DistanceSquared(PointF a, PointF b)
		{
			float dx = a.X - b.X;
			float dy = a.Y - b.Y;
			return dx * dx + dy * dy;
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		PointF GetBasePoint()
		{
			const int drawAreaTop = 180;
			const int drawAreaHeight = this->ClientSize.Height - 180;
			const int leftWallX = this->ClientSize.Width * 25 / 100;
			return PointF(static_cast<float>(leftWallX), static_cast<float>(drawAreaTop + drawAreaHeight / 2));
		}

		void InitGraphicsResources()
		{
			uiFontBold11 = gcnew System::Drawing::Font("Yu Gothic UI", 11, FontStyle::Bold);
			uiFontBold10 = gcnew System::Drawing::Font("Yu Gothic UI", 10, FontStyle::Bold);
			uiFontTextBox = gcnew System::Drawing::Font("Yu Gothic UI", 11, FontStyle::Bold);
			wallPen = gcnew Pen(Color::Black, 2.0f);
			dashedPen = gcnew Pen(Color::Black, 2.0f);
			dashedPen->DashStyle = DashStyle::Dash;
			targetPen = gcnew Pen(Color::Green, 3.0f);
			targetPen->DashStyle = DashStyle::Dot;
			startPen = gcnew Pen(Color::FromArgb(255, 140, 0), 3.0f);
			startPen->DashStyle = DashStyle::Dot;
			pathPen = gcnew Pen(Color::FromArgb(40, 90, 180), 2.5f);
			pathPen->DashStyle = DashStyle::Dash;
			penRod = gcnew Pen(Color::Red, 6.0f);
			jointBrush = gcnew SolidBrush(Color::Blue);
			waypointBrush = gcnew SolidBrush(Color::FromArgb(40, 90, 180));
			wallHatchBrush = gcnew HatchBrush(HatchStyle::BackwardDiagonal, Color::LightGray, Color::White);
			obstaclePen = gcnew Pen(Color::FromArgb(90, 30, 30), 2.0f);
			obstacleMarginPen = gcnew Pen(Color::FromArgb(215, 140, 0), 2.0f);
			obstacleMarginPen->DashStyle = DashStyle::Dash;
			obstacleBrush = gcnew SolidBrush(Color::FromArgb(180, 120, 120, 120));
		}

		void InitAnimation()
		{
			animationTimer = gcnew Timer();
			animationTimer->Interval = AnimationIntervalMs;
			animationTimer->Tick += gcnew EventHandler(this, &MyForm::OnAnimationTick);
		}

		void InitUI()
		{
			cbDemoMode = gcnew ComboBox();
			cbDemoMode->Location = Point(920, 20);
			cbDemoMode->Width = 260;
			cbDemoMode->Height = 28;
			cbDemoMode->DropDownStyle = ComboBoxStyle::DropDownList;
			cbDemoMode->Font = uiFontBold11;
			cbDemoMode->BackColor = SystemColors::Info;
			cbDemoMode->FlatStyle = FlatStyle::Flat;
			cbDemoMode->Items->Add(L"Позиционирование");
			cbDemoMode->Items->Add(L"Планирование траектории");
			cbDemoMode->SelectedIndex = 0;
			cbDemoMode->SelectedIndexChanged += gcnew EventHandler(this, &MyForm::OnDemoModeChanged);
			this->Controls->Add(cbDemoMode);

			cbBackend = gcnew ComboBox();
			cbBackend->Location = Point(920, 54);
			cbBackend->Width = 260;
			cbBackend->Height = 28;
			cbBackend->DropDownStyle = ComboBoxStyle::DropDownList;
			cbBackend->Font = uiFontBold11;
			cbBackend->BackColor = SystemColors::Info;
			cbBackend->FlatStyle = FlatStyle::Flat;
			cbBackend->Items->Add(L"AGP");
			cbBackend->Items->Add(L"TRAC-IK");
			cbBackend->SelectedIndex = 0;
			cbBackend->SelectedIndexChanged += gcnew EventHandler(this, &MyForm::OnBackendChanged);
			this->Controls->Add(cbBackend);

			Label^ L = gcnew Label();
			L->Text = L"Макс. угол (рад.)";
			L->Location = Point(20, 20);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudMaxTheta = gcnew NumericUpDown();
			nudMaxTheta->Location = Point(20, 52);
			nudMaxTheta->Width = 200;
			nudMaxTheta->DecimalPlaces = 3;
			nudMaxTheta->Minimum = Decimal(1) / Decimal(100);
			nudMaxTheta->Maximum = Decimal(314159) / Decimal(100000);
			nudMaxTheta->Value = Decimal(200) / Decimal(100);
			nudMaxTheta->Font = uiFontTextBox;
			nudMaxTheta->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(nudMaxTheta);

			L = gcnew Label();
			L->Text = L"Базовая длина";
			L->Location = Point(245, 20);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudBaseLength = gcnew NumericUpDown();
			nudBaseLength->Location = Point(245, 52);
			nudBaseLength->Width = 200;
			nudBaseLength->DecimalPlaces = 2;
			nudBaseLength->Minimum = Decimal(1) / Decimal(2);
			nudBaseLength->Maximum = Decimal(200) / Decimal(100);
			nudBaseLength->Value = Decimal(100) / Decimal(100);
			nudBaseLength->Font = uiFontTextBox;
			nudBaseLength->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(nudBaseLength);

			L = gcnew Label();
			L->Text = L"Макс. коэфф. растяжения/сжатия";
			L->Location = Point(470, 20);
			L->Width = 300;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudStretchFactor = gcnew NumericUpDown();
			nudStretchFactor->Location = Point(470, 52);
			nudStretchFactor->Width = 200;
			nudStretchFactor->DecimalPlaces = 2;
			nudStretchFactor->Minimum = Decimal(100) / Decimal(100);
			nudStretchFactor->Maximum = Decimal(150) / Decimal(100);
			nudStretchFactor->Increment = Decimal(1) / Decimal(100);
			nudStretchFactor->Value = Decimal(150) / Decimal(100);
			nudStretchFactor->Font = uiFontTextBox;
			nudStretchFactor->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(nudStretchFactor);

			cbVarLen = gcnew CheckBox();
			cbVarLen->Text = L"Переменные длины";
			cbVarLen->Location = Point(695, 52);
			cbVarLen->Width = 200;
			cbVarLen->Checked = false;
			cbVarLen->Font = uiFontBold11;
			cbVarLen->CheckedChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(cbVarLen);

			L = gcnew Label();
			L->Text = L"Цель X";
			L->Location = Point(20, 107);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudTargetX = gcnew NumericUpDown();
			nudTargetX->Location = Point(20, 139);
			nudTargetX->Width = 200;
			nudTargetX->DecimalPlaces = 2;
			nudTargetX->Minimum = Decimal(-100) / Decimal(10);
			nudTargetX->Maximum = Decimal(100) / Decimal(10);
			nudTargetX->Value = Decimal(25) / Decimal(10);
			nudTargetX->Font = uiFontTextBox;
			nudTargetX->ValueChanged += gcnew EventHandler(this, &MyForm::OnTargetChanged);
			this->Controls->Add(nudTargetX);

			L = gcnew Label();
			L->Text = L"Цель Y";
			L->Location = Point(245, 107);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudTargetY = gcnew NumericUpDown();
			nudTargetY->Location = Point(245, 139);
			nudTargetY->Width = 200;
			nudTargetY->DecimalPlaces = 2;
			nudTargetY->Minimum = Decimal(-100) / Decimal(10);
			nudTargetY->Maximum = Decimal(100) / Decimal(10);
			nudTargetY->Value = Decimal(-10) / Decimal(10);
			nudTargetY->Font = uiFontTextBox;
			nudTargetY->ValueChanged += gcnew EventHandler(this, &MyForm::OnTargetChanged);
			this->Controls->Add(nudTargetY);

			L = gcnew Label();
			L->Text = L"Надежность (r)";
			L->Location = Point(470, 107);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudR = gcnew NumericUpDown();
			nudR->Location = Point(470, 139);
			nudR->Width = 200;
			nudR->DecimalPlaces = 2;
			nudR->Minimum = Decimal(100) / Decimal(100);
			nudR->Maximum = Decimal(2000) / Decimal(100);
			nudR->Value = Decimal(105) / Decimal(100);
			nudR->Font = uiFontTextBox;
			nudR->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(nudR);

			cbAdaptive = gcnew CheckBox();
			cbAdaptive->Text = L"Адаптивная схема";
			cbAdaptive->Location = Point(695, 139);
			cbAdaptive->Width = 200;
			cbAdaptive->Checked = true;
			cbAdaptive->Font = uiFontBold11;
			cbAdaptive->CheckedChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(cbAdaptive);

			L = gcnew Label();
			L->Text = L"Точность";
			L->Location = Point(20, 194);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			currentEps = 1e-9f;
			txtEps = gcnew TextBox();
			txtEps->Location = Point(20, 226);
			txtEps->Width = 120;
			txtEps->Font = uiFontTextBox;
			txtEps->Text = L"1E-09";
			txtEps->TextChanged += gcnew EventHandler(this, &MyForm::OnEpsTextChanged);
			this->Controls->Add(txtEps);

			Button^ btnEpsUp = gcnew Button();
			btnEpsUp->Text = L"×10";
			btnEpsUp->Location = Point(145, 226);
			btnEpsUp->Width = 32;
			btnEpsUp->Height = 26;
			btnEpsUp->Font = uiFontBold11;
			btnEpsUp->TextAlign = ContentAlignment::TopRight;
			btnEpsUp->Padding = System::Windows::Forms::Padding(0, 0, 2, 3);
			btnEpsUp->Click += gcnew EventHandler(this, &MyForm::OnEpsOrderUp);
			this->Controls->Add(btnEpsUp);

			Button^ btnEpsDown = gcnew Button();
			btnEpsDown->Text = L"÷10";
			btnEpsDown->Location = Point(181, 226);
			btnEpsDown->Width = 32;
			btnEpsDown->Height = 26;
			btnEpsDown->Font = uiFontBold11;
			btnEpsDown->TextAlign = ContentAlignment::TopRight;
			btnEpsDown->Padding = System::Windows::Forms::Padding(0, 0, 2, 3);
			btnEpsDown->Click += gcnew EventHandler(this, &MyForm::OnEpsOrderDown);
			this->Controls->Add(btnEpsDown);

			L = gcnew Label();
			L->Text = L"Макс. итераций";
			L->Location = Point(245, 194);
			L->Width = 200;
			L->Font = uiFontBold11;
			this->Controls->Add(L);

			nudMaxIter = gcnew NumericUpDown();
			nudMaxIter->Location = Point(245, 226);
			nudMaxIter->Width = 200;
			nudMaxIter->Minimum = 10;
			nudMaxIter->Maximum = 500000;
			nudMaxIter->Value = 1000;
			nudMaxIter->Font = uiFontTextBox;
			nudMaxIter->Increment = 100;
			nudMaxIter->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged);
			this->Controls->Add(nudMaxIter);

			btnAdd = gcnew Button();
			btnAdd->Text = L"+ Звено";
			btnAdd->Location = Point(465, 191);
			btnAdd->Width = 90;
			btnAdd->Height = 35;
			btnAdd->BackColor = SystemColors::Info;
			btnAdd->Cursor = Cursors::Hand;
			btnAdd->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64);
			btnAdd->FlatAppearance->BorderSize = 3;
			btnAdd->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255);
			btnAdd->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255);
			btnAdd->FlatStyle = FlatStyle::Flat;
			btnAdd->Font = uiFontBold11;
			btnAdd->ForeColor = SystemColors::ControlDarkDark;
			btnAdd->Click += gcnew EventHandler(this, &MyForm::OnAddClick);
			this->Controls->Add(btnAdd);

			btnRem = gcnew Button();
			btnRem->Text = L"- Звено";
			btnRem->Location = Point(560, 191);
			btnRem->Width = 90;
			btnRem->Height = 35;
			btnRem->BackColor = SystemColors::Info;
			btnRem->Cursor = Cursors::Hand;
			btnRem->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64);
			btnRem->FlatAppearance->BorderSize = 3;
			btnRem->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255);
			btnRem->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255);
			btnRem->FlatStyle = FlatStyle::Flat;
			btnRem->Font = uiFontBold11;
			btnRem->ForeColor = SystemColors::ControlDarkDark;
			btnRem->Click += gcnew EventHandler(this, &MyForm::OnRemClick);
			this->Controls->Add(btnRem);

			btnOptimize = gcnew Button();
			btnOptimize->Text = L"Оптимизировать";
			btnOptimize->Location = Point(680, 191);
			btnOptimize->Width = 150;
			btnOptimize->Height = 35;
			btnOptimize->BackColor = SystemColors::Info;
			btnOptimize->Cursor = Cursors::Hand;
			btnOptimize->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64);
			btnOptimize->FlatAppearance->BorderSize = 3;
			btnOptimize->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255);
			btnOptimize->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255);
			btnOptimize->FlatStyle = FlatStyle::Flat;
			btnOptimize->Font = uiFontBold11;
			btnOptimize->ForeColor = SystemColors::ControlDarkDark;
			btnOptimize->Click += gcnew EventHandler(this, &MyForm::OnOptimizeClick);
			this->Controls->Add(btnOptimize);

			btnGenerateObstacles = gcnew Button();
			btnGenerateObstacles->Location = Point(465, 237);
			btnGenerateObstacles->Width = 365;
			btnGenerateObstacles->Height = 35;
			btnGenerateObstacles->BackColor = SystemColors::Info;
			btnGenerateObstacles->Cursor = Cursors::Hand;
			btnGenerateObstacles->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64);
			btnGenerateObstacles->FlatAppearance->BorderSize = 3;
			btnGenerateObstacles->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255);
			btnGenerateObstacles->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255);
			btnGenerateObstacles->FlatStyle = FlatStyle::Flat;
			btnGenerateObstacles->Font = uiFontBold11;
			btnGenerateObstacles->ForeColor = SystemColors::ControlDarkDark;
			btnGenerateObstacles->Click += gcnew EventHandler(this, &MyForm::OnGenerateObstaclesClick);
			this->Controls->Add(btnGenerateObstacles);

			btnClearObstacles = gcnew Button();
			btnClearObstacles->Location = Point(465, 284);
			btnClearObstacles->Width = 365;
			btnClearObstacles->Height = 35;
			btnClearObstacles->BackColor = SystemColors::Info;
			btnClearObstacles->Cursor = Cursors::Hand;
			btnClearObstacles->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64);
			btnClearObstacles->FlatAppearance->BorderSize = 3;
			btnClearObstacles->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255);
			btnClearObstacles->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255);
			btnClearObstacles->FlatStyle = FlatStyle::Flat;
			btnClearObstacles->Font = uiFontBold11;
			btnClearObstacles->ForeColor = SystemColors::ControlDarkDark;
			btnClearObstacles->Text = L"Очистить";
			btnClearObstacles->Click += gcnew EventHandler(this, &MyForm::OnClearObstaclesClick);
			this->Controls->Add(btnClearObstacles);

			lblInfo = gcnew Label();
			lblInfo->Location = Point(835, 194);
			lblInfo->Size = System::Drawing::Size(275, 125);
			lblInfo->BorderStyle = BorderStyle::FixedSingle;
			lblInfo->Font = uiFontBold10;
			this->Controls->Add(lblInfo);

			lblStartX = gcnew Label();
			lblStartX->Text = L"Начало X";
			lblStartX->Location = Point(20, 272);
			lblStartX->Width = 200;
			lblStartX->Font = uiFontBold11;
			this->Controls->Add(lblStartX);

			nudStartX = gcnew NumericUpDown();
			nudStartX->Location = Point(20, 304);
			nudStartX->Width = 200;
			nudStartX->DecimalPlaces = 2;
			nudStartX->Minimum = Decimal(-100) / Decimal(10);
			nudStartX->Maximum = Decimal(100) / Decimal(10);
			nudStartX->Value = Decimal(125) / Decimal(100);
			nudStartX->Font = uiFontTextBox;
			nudStartX->ValueChanged += gcnew EventHandler(this, &MyForm::OnStartPointChanged);
			this->Controls->Add(nudStartX);

			lblStartY = gcnew Label();
			lblStartY->Text = L"Начало Y";
			lblStartY->Location = Point(245, 272);
			lblStartY->Width = 200;
			lblStartY->Font = uiFontBold11;
			this->Controls->Add(lblStartY);

			nudStartY = gcnew NumericUpDown();
			nudStartY->Location = Point(245, 304);
			nudStartY->Width = 200;
			nudStartY->DecimalPlaces = 2;
			nudStartY->Minimum = Decimal(-100) / Decimal(10);
			nudStartY->Maximum = Decimal(100) / Decimal(10);
			nudStartY->Value = Decimal(-50) / Decimal(100);
			nudStartY->Font = uiFontTextBox;
			nudStartY->ValueChanged += gcnew EventHandler(this, &MyForm::OnStartPointChanged);
			this->Controls->Add(nudStartY);

			UpdateBackendUiState();
		}

		void UpdateBackendUiState()
		{
			bool trajectoryMode = IsTrajectoryMode();
			cbBackend->Enabled = true;

			bool tracIkSelected = (cbBackend->SelectedIndex == 1);
			bool obstaclesDisabled = trajectoryMode && tracIkSelected;
			bool obstaclesEnabled = !obstaclesDisabled;

			btnGenerateObstacles->Enabled = obstaclesEnabled;
			if (obstaclesDisabled)
			{
				btnGenerateObstacles->Text = L"Препятствия отключены";
				btnGenerateObstacles->ForeColor = Color::Gold;
				btnGenerateObstacles->BackColor = SystemColors::Control;
			}
			else
			{
				btnGenerateObstacles->Text = L"Сгенерировать препятствия";
				btnGenerateObstacles->ForeColor = SystemColors::ControlDarkDark;
				btnGenerateObstacles->BackColor = SystemColors::Info;
			}

			btnClearObstacles->Enabled = obstaclesEnabled;
			if (obstaclesEnabled)
			{
				btnClearObstacles->BackColor = SystemColors::Info;
			}
			else
			{
				btnClearObstacles->BackColor = SystemColors::Control;
			}

			if (obstaclesDisabled && obstacleX->Count > 0)
			{
				obstacleX->Clear();
				obstacleY->Clear();
				obstacleHalf->Clear();
				ClearTrajectoryCache();
				this->Invalidate();
			}
		}

		void ResetRandomConfig()
		{
			nSegments = 1;
			angles->Clear();
			lengths->Clear();
			obstacleX->Clear();
			obstacleY->Clear();
			obstacleHalf->Clear();
			angles->Add(0.0f);
			lengths->Add(static_cast<float>(nudBaseLength->Value));
			variableLengths = false;
			ClearTrajectoryCache();
			this->Invalidate();
		}

		void ClearTrajectoryCache()
		{
			StopAnimation();
			plannedPoses->Clear();
			animationFrames->Clear();
			trajectoryPathWorld->Clear();
		}

		void StopAnimation()
		{
			animationTimer->Stop();
			animationFrameIndex = 0;
			animationRunning = false;
		}

		void ClearObstacles()
		{
			obstacleX->Clear();
			obstacleY->Clear();
			obstacleHalf->Clear();
			ClearTrajectoryCache();
			this->Invalidate();
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		float Rand01()
		{
			rngState ^= rngState << 13;
			rngState ^= rngState >> 17;
			rngState ^= rngState << 5;
			return static_cast<float>(static_cast<unsigned int>(rngState)) * 2.3283064e-10f;
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		float Lerp(float a, float b, float t)
		{
			return a + (b - a) * t;
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		bool IsTrajectoryMode()
		{
			return demoMode == DemoMode::TrajectoryPlanning;
		}

		void UpdateTrajectoryUiState()
		{
			bool trajectoryMode = IsTrajectoryMode();
			lblStartX->Visible = trajectoryMode;
			lblStartY->Visible = trajectoryMode;
			nudStartX->Visible = trajectoryMode;
			nudStartY->Visible = trajectoryMode;
			lblStartX->Enabled = trajectoryMode;
			lblStartY->Enabled = trajectoryMode;
			nudStartX->Enabled = trajectoryMode;
			nudStartY->Enabled = trajectoryMode;
			UpdateBackendUiState();
			if (!trajectoryMode) ClearTrajectoryCache();
			this->Invalidate();
		}

		void SetDefaultStartPoint(bool forceResetCustomization)
		{
			if (forceResetCustomization) startPointCustomized = false;
			if (startPointCustomized) return;
			syncingDefaultStartPoint = true;
			nudStartX->Value = Decimal(static_cast<double>(static_cast<float>(nudTargetX->Value) * 0.5f));
			nudStartY->Value = Decimal(static_cast<double>(static_cast<float>(nudTargetY->Value) * 0.5f));
			syncingDefaultStartPoint = false;
		}

		array<float>^ BuildObstacleBuffer()
		{
			int count = obstacleX->Count;
			array<float>^ data = gcnew array<float>(count * 3);
			for (int i = 0; i < count; ++i)
			{
				data[3 * i + 0] = obstacleX[i];
				data[3 * i + 1] = obstacleY[i];
				data[3 * i + 2] = obstacleHalf[i];
			}
			return data;
		}

		void GenerateRandomObstacles()
		{
			obstacleX->Clear();
			obstacleY->Clear();
			obstacleHalf->Clear();

			bool varLen = cbVarLen->Checked;
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float stretchFactor = static_cast<float>(nudStretchFactor->Value);
			float tx = static_cast<float>(nudTargetX->Value);
			float ty = static_cast<float>(nudTargetY->Value);
			float dist = static_cast<float>(Math::Sqrt(tx * tx + ty * ty));
			float maxReach = static_cast<float>(nSegments) * (varLen ? baseLength * stretchFactor : baseLength);
			float slack = maxReach - dist;

			float halfMin = 0.14f * baseLength;
			if (halfMin < 0.12f) halfMin = 0.12f;
			float halfMax = 0.26f * baseLength + 0.05f * (slack / (baseLength + 1e-6f));
			if (halfMax > 0.32f) halfMax = 0.32f;
			if (halfMax < halfMin + 0.03f) halfMax = halfMin + 0.03f;

			float alongMargin = 0.55f * baseLength;
			float projectionMargin = 1.5f * (halfMax + ObstacleClearance);
			if (alongMargin < projectionMargin) alongMargin = projectionMargin;
			if (alongMargin < 0.35f) alongMargin = 0.35f;

			float usableLen = dist - 2.0f * alongMargin;
			int obstacleCountToCreate = 2 + static_cast<int>(Rand01() * 3.0f);
			if (nSegments == 2 && obstacleCountToCreate > 3) obstacleCountToCreate = 3;
			if (slack < 0.35f * baseLength && obstacleCountToCreate > 2) obstacleCountToCreate = 2;
			while (obstacleCountToCreate > 2)
			{
				float gapTest = usableLen / static_cast<float>(obstacleCountToCreate + 1);
				if (gapTest >= 2.2f * halfMax) break;
				--obstacleCountToCreate;
			}

			float ux = tx / dist;
			float uy = ty / dist;
			float nx = -uy;
			float ny = ux;
			float gap = usableLen / static_cast<float>(obstacleCountToCreate + 1);
			float firstSide = (Rand01() < 0.5f) ? -1.0f : 1.0f;

			for (int i = 0; i < obstacleCountToCreate; ++i)
			{
				float nominalS = alongMargin + gap * static_cast<float>(i + 1);
				float jitter = (Rand01() - 0.5f) * gap * 0.18f;
				float s = nominalS + jitter;
				float half = halfMin + (halfMax - halfMin) * Rand01();
				float side = ((i & 1) == 0) ? firstSide : -firstSide;
				float offsetMag = half * (0.18f + 0.28f * Rand01());
				obstacleX->Add(ux * s + nx * side * offsetMag);
				obstacleY->Add(uy * s + ny * side * offsetMag);
				obstacleHalf->Add(half);
			}

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void UpdateEpsDisplay()
		{
			txtEps->Text = currentEps.ToString("E3", System::Globalization::CultureInfo::InvariantCulture);
		}

		void OnEpsTextChanged(System::Object^ sender, System::EventArgs^ e)
		{
			String^ text = txtEps->Text;
			float val;
			if (float::TryParse(text, System::Globalization::NumberStyles::Float, System::Globalization::CultureInfo::InvariantCulture, val))
			{
				if (val < 1e-9f) val = 1e-9f;
				if (val > 1e-1f) val = 1e-1f;
				currentEps = val;
				UpdateEpsDisplay();
				OnAnyChanged(nullptr, nullptr);
			}
			else
			{
				UpdateEpsDisplay();
			}
		}

		void OnEpsOrderUp(System::Object^ sender, System::EventArgs^ e)
		{
			float newVal = currentEps * 10.0f;
			if (newVal > 1e-1f) newVal = 1e-1f;
			currentEps = newVal;
			UpdateEpsDisplay();
			OnAnyChanged(nullptr, nullptr);
		}

		void OnEpsOrderDown(System::Object^ sender, System::EventArgs^ e)
		{
			float newVal = currentEps / 10.0f;
			if (newVal < 1e-9f) newVal = 1e-9f;
			currentEps = newVal;
			UpdateEpsDisplay();
			OnAnyChanged(nullptr, nullptr);
		}

		System::Void OnClearObstaclesClick(System::Object^ sender, System::EventArgs^ e)
		{
			ClearObstacles();
		}

		System::Void OnGenerateObstaclesClick(System::Object^ sender, System::EventArgs^ e)
		{
			GenerateRandomObstacles();
		}

		void ApplyPoseToManipulator(PoseSnapshot^ pose)
		{
			angles->Clear();
			lengths->Clear();
			for (int i = 0; i < nSegments; ++i)
			{
				angles->Add(pose->Angles[i]);
				lengths->Add(pose->Lengths[i]);
			}
		}

		void ComputeEndEffector(array<float>^ poseAngles, array<float>^ poseLengths, float% outX, float% outY)
		{
			float x = 0.0f;
			float y = 0.0f;
			float phi = PI_2;
			for (int i = 0; i < poseAngles->Length; ++i)
			{
				phi += poseAngles[i];
				x += poseLengths[i] * static_cast<float>(Math::Cos(static_cast<double>(phi)));
				y += poseLengths[i] * static_cast<float>(Math::Sin(static_cast<double>(phi)));
			}
			outX = x;
			outY = y;
		}

		PoseSnapshot^ RunAgpCppAtPoint(float tx, float ty)
		{
			variableLengths = cbVarLen->Checked;
			float maxTheta = static_cast<float>(nudMaxTheta->Value);
			int maxIter = static_cast<int>(nudMaxIter->Value);
			bool adaptive = cbAdaptive->Checked;
			float r_param = static_cast<float>(nudR->Value);
			float eps = currentEps;
			unsigned int seed = static_cast<unsigned int>(GetTickCount());
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float stretchFactor = static_cast<float>(nudStretchFactor->Value);

			array<float>^ obstacleData = BuildObstacleBuffer();
			pin_ptr<float> pinnedObstacles = nullptr;
			const float* pObstacleData = nullptr;
			if (obstacleData->Length > 0)
			{
				pinnedObstacles = &obstacleData[0];
				pObstacleData = pinnedObstacles;
			}
			int obstacleCount = obstacleX->Count;

			pStart(
				nSegments, variableLengths, maxTheta, tx, ty, maxIter,
				r_param, adaptive, eps, seed, baseLength, stretchFactor,
				pObstacleData, obstacleCount, 0, nullptr, nullptr
			);

			LARGE_INTEGER t0, t1, fq;
			QueryPerformanceCounter(&t0);

			float* bestQ = nullptr;
			float bestX = 0.0f;
			float bestY = 0.0f;
			float bestF = 0.0f;
			size_t actualIterations = 0u;
			float achievedEps = 0.0f;

			fManip(
				nSegments, variableLengths, maxTheta, tx, ty, maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObstacleData, obstacleCount,
				&bestQ, &bestX, &bestY, &bestF, &actualIterations, &achievedEps,
				0, nullptr
			);

			QueryPerformanceCounter(&t1);
			QueryPerformanceFrequency(&fq);

			float micros = static_cast<float>(1e6 * static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(fq.QuadPart));

			PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);
			for (int i = 0; i < nSegments; ++i) pose->Angles[i] = bestQ[i];
			if (variableLengths)
			{
				for (int i = 0; i < nSegments; ++i) pose->Lengths[i] = bestQ[nSegments + i];
			}
			else
			{
				for (int i = 0; i < nSegments; ++i) pose->Lengths[i] = baseLength;
			}

			pFree(bestQ);

			pose->EndX = bestX;
			pose->EndY = bestY;
			pose->BestF = bestF;
			pose->Iterations = static_cast<int>(actualIterations);
			pose->AchievedEps = achievedEps;
			pose->Micros = micros;
			return pose;
		}

		PoseSnapshot^ RunTracIkPositioning(float tx, float ty)
		{
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float maxTheta = static_cast<float>(nudMaxTheta->Value);
			float eps = currentEps;

			KDL::Chain chain;
			for (int i = 0; i < nSegments; ++i) {
				chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ),
					KDL::Frame(KDL::Vector(baseLength, 0.0, 0.0))));
			}
			int nJoints = chain.getNrOfJoints();
			KDL::JntArray q_min(nJoints);
			KDL::JntArray q_max(nJoints);
			for (int i = 0; i < nJoints; ++i) {
				if (i == 0) {
					q_min(i) = -maxTheta + PI_2;
					q_max(i) = maxTheta + PI_2;
				}
				else {
					q_min(i) = -maxTheta;
					q_max(i) = 0.0f;
				}
			}
			double max_time = 1.0;
			trac_ik::TRAC_IK ik_solver(chain, q_min, q_max, max_time, eps, trac_ik::SolveType::Speed);
			KDL::Frame target(KDL::Rotation::Identity(), KDL::Vector(tx, ty, 0.0));

			KDL::JntArray q_init(nJoints);
			for (int i = 0; i < nJoints; ++i)
			{
				if (i == 0) {
					q_init(i) = PI_2;
				}
				else {
					q_init(i) = 0.0f;
				}
			}

			KDL::JntArray q_out(nJoints);
			KDL::Twist tolerances(KDL::Vector(eps, eps, 0.0), KDL::Vector(0.0, 0.0, 0.0));
			ik_solver.CartToJnt(q_init, target, q_out, tolerances);

			PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);
			for (int i = 0; i < nSegments; ++i) {
				if (i == 0) {
					pose->Angles[i] = static_cast<float>(q_out(0) - PI_2);
				}
				else {
					pose->Angles[i] = static_cast<float>(q_out(i));
				}
				pose->Lengths[i] = baseLength;
			}
			ComputeEndEffector(pose->Angles, pose->Lengths, pose->EndX, pose->EndY);
			float dx = pose->EndX - tx;
			float dy = pose->EndY - ty;
			pose->BestF = sqrtf(dx * dx + dy * dy);
			pose->Iterations = 1;
			pose->AchievedEps = 0.0f;
			pose->Micros = 0.0f;
			return pose;
		}

		void UpdatePositioningStats(PoseSnapshot^ pose, float tx, float ty)
		{
			float dx = pose->EndX - tx;
			float dy = pose->EndY - ty;
			float distance = sqrtf(dx * dx + dy * dy);
			lblInfo->Text = String::Format(
				L"Функционал: {0:F6}\n"
				L"Близость захвата: {1:F5}\n"
				L"Точка: ({2:F3}, {3:F3})\n"
				L"Время: {4:F2} мс\n"
				L"Число шагов: {5}\n"
				L"Достигнутая точность: {6:E3}",
				pose->BestF,
				distance,
				pose->EndX,
				pose->EndY,
				pose->Micros / 1000.0f,
				pose->Iterations,
				pose->AchievedEps
			);
		}

		void UpdateTrajectoryStats(List<PoseSnapshot^>^ poses, float totalMicros, size_t totalIterations, float finalAchievedEps)
		{
			float totalEnergy = 0.0f;
			for (int i = 1; i < poses->Count; ++i)
			{
				totalEnergy += ComputeTransitionEnergy(poses[i - 1], poses[i]);
			}

			PoseSnapshot^ lastPose = poses[poses->Count - 1];
			float rawTargetX = static_cast<float>(nudTargetX->Value);
			float rawTargetY = static_cast<float>(nudTargetY->Value);
			float finalDx = lastPose->EndX - rawTargetX;
			float finalDy = lastPose->EndY - rawTargetY;
			float finalDistance = sqrtf(finalDx * finalDx + finalDy * finalDy);
			int actualIntermediateCount = poses->Count - 2;

			lblInfo->Text = String::Format(
				L"Промежуточных точек: {0}\n"
				L"Функционал траектории: {1:F6}\n"
				L"Близость финиша: {2:F5}\n"
				L"Время: {3:F2} мс\n"
				L"Число шагов: {4}\n"
				L"Достигнутая точность: {5:E3}",
				actualIntermediateCount,
				totalEnergy,
				finalDistance,
				totalMicros / 1000.0f,
				totalIterations,
				finalAchievedEps
			);
		}

		float ComputeTransitionEnergy(PoseSnapshot^ prevPose, PoseSnapshot^ nextPose)
		{
			float total = 0.0f;
			float prevPrefix = 0.0f;
			float nextPrefix = 0.0f;
			for (int i = 0; i < prevPose->Angles->Length; ++i)
			{
				float d = WrappedDelta(prevPose->Angles[i], nextPose->Angles[i]);
				total += d * d;
				prevPrefix += prevPose->Angles[i];
				nextPrefix += nextPose->Angles[i];
				float dp = WrappedDelta(prevPrefix, nextPrefix);
				total += TransitionPrefixEnergyWeight * dp * dp;
			}
			if (variableLengths)
			{
				for (int i = 0; i < prevPose->Lengths->Length; ++i)
				{
					float dl = nextPose->Lengths[i] - prevPose->Lengths[i];
					total += TransitionLengthEnergyWeight * dl * dl;
				}
			}
			return total;
		}

		void RunTrajectoryPlanningMode()
		{
			if (cbBackend->SelectedIndex == 1)
			{
				RunTrajectoryWithTracIk();
			}
			else
			{
				RunTrajectoryWithAGP();
			}
		}

		void RunTrajectoryWithAGP()
		{
			StopAnimation();
			ClearTrajectoryCache();

			array<float>^ obsData = BuildObstacleBuffer();
			pin_ptr<float> pinnedObs = nullptr;
			const float* pObs = nullptr;
			if (obsData->Length > 0)
			{
				pinnedObs = &obsData[0];
				pObs = pinnedObs;
			}
			int obstacleCount = obstacleX->Count;

			PointF startPoint(static_cast<float>(nudStartX->Value), static_cast<float>(nudStartY->Value));
			PointF targetPoint(static_cast<float>(nudTargetX->Value), static_cast<float>(nudTargetY->Value));
			float maxTheta = static_cast<float>(nudMaxTheta->Value);
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float stretchFactor = static_cast<float>(nudStretchFactor->Value);
			variableLengths = cbVarLen->Checked;
			bool varLen = variableLengths;
			int maxIter = static_cast<int>(nudMaxIter->Value);
			float r_param = static_cast<float>(nudR->Value);
			bool adaptive = cbAdaptive->Checked;
			float eps = currentEps;
			unsigned int seed = static_cast<unsigned int>(GetTickCount());
			int stateDim = nSegments << 1;

			pStart(
				nSegments, varLen, maxTheta,
				startPoint.X, startPoint.Y,
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObs, obstacleCount,
				0, nullptr, nullptr
			);

			LARGE_INTEGER startT0, startT1, startFq;
			QueryPerformanceCounter(&startT0);

			float* startQ = nullptr;
			float startX = 0.0f;
			float startY = 0.0f;
			float startF = 0.0f;
			size_t startIterations = 0u;
			float startEps = 0.0f;

			fManip(
				nSegments, varLen, maxTheta, startPoint.X, startPoint.Y,
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObs, obstacleCount,
				&startQ, &startX, &startY, &startF, &startIterations, &startEps,
				0, nullptr
			);

			QueryPerformanceCounter(&startT1);
			QueryPerformanceFrequency(&startFq);

			PoseSnapshot^ startPose = gcnew PoseSnapshot(nSegments);
			for (int i = 0; i < nSegments; ++i)
			{
				startPose->Angles[i] = startQ[i];
				startPose->Lengths[i] = varLen ? startQ[nSegments + i] : baseLength;
			}
			pFree(startQ);

			startPose->EndX = startX;
			startPose->EndY = startY;
			startPose->BestF = startF;
			startPose->Iterations = static_cast<int>(startIterations);
			startPose->AchievedEps = startEps;
			startPose->Micros = static_cast<float>(1e6 * static_cast<double>(startT1.QuadPart - startT0.QuadPart) / static_cast<double>(startFq.QuadPart));

			std::vector<float> startState(stateDim);
			for (int i = 0; i < nSegments; ++i)
			{
				startState[i] = startPose->Angles[i];
				startState[nSegments + i] = startPose->Lengths[i];
			}

			pStart(
				nSegments, varLen, maxTheta,
				targetPoint.X, targetPoint.Y,
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObs, obstacleCount,
				1, startState.data(), nullptr
			);

			LARGE_INTEGER t0, t1, fq;
			QueryPerformanceCounter(&t0);

			float* finalQ = nullptr;
			float finalX = 0.0f;
			float finalY = 0.0f;
			float finalF = 0.0f;
			size_t finalIterations = 0;
			float finalEps = 0.0f;

			fManip(
				nSegments, varLen, maxTheta, targetPoint.X, targetPoint.Y,
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObs, obstacleCount,
				&finalQ, &finalX, &finalY, &finalF, &finalIterations, &finalEps,
				1, startState.data()
			);

			std::vector<float> finalState(stateDim);
			for (int i = 0; i < nSegments; ++i)
			{
				finalState[i] = finalQ[i];
				finalState[nSegments + i] = varLen ? finalQ[nSegments + i] : baseLength;
			}
			pFree(finalQ);

			pStart(
				nSegments, varLen, maxTheta, 0.0f, 0.0f, maxIter,
				r_param, adaptive, eps, seed, baseLength, stretchFactor,
				pObs, obstacleCount,
				2, startState.data(), finalState.data()
			);

			float* trajPoints = nullptr;
			int pointCount = 0;
			size_t totalIterations = 0u;

			pBuildTrajectory(
				nSegments, varLen, maxTheta,
				startState.data(), finalState.data(),
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				pObs, obstacleCount,
				&trajPoints, &pointCount, &totalIterations
			);
			totalIterations += static_cast<size_t>(startPose->Iterations) + finalIterations;

			QueryPerformanceCounter(&t1);
			QueryPerformanceFrequency(&fq);
			float micros = static_cast<float>(1e6 * static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(fq.QuadPart));

			plannedPoses->Clear();
			for (int i = 0; i < pointCount; ++i)
			{
				float* ptr = trajPoints + i * stateDim;
				PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);
				for (int j = 0; j < nSegments; ++j)
				{
					pose->Angles[j] = ptr[j];
					pose->Lengths[j] = varLen ? ptr[nSegments + j] : baseLength;
				}
				ComputeEndEffector(pose->Angles, pose->Lengths, pose->EndX, pose->EndY);
				plannedPoses->Add(pose);
			}
			pFree(trajPoints);

			UpdateTrajectoryStats(plannedPoses, micros, totalIterations, finalEps);
			BuildAnimationFramesFromPlan();
			ApplyPoseToManipulator(plannedPoses[0]);
			StartAnimationIfNeeded();
			this->Invalidate();
		}

		void RunTrajectoryWithTracIk()
		{
			StopAnimation();
			ClearTrajectoryCache();

			PointF startPoint(static_cast<float>(nudStartX->Value), static_cast<float>(nudStartY->Value));
			PointF targetPoint(static_cast<float>(nudTargetX->Value), static_cast<float>(nudTargetY->Value));

			PoseSnapshot^ startPose = RunTracIkPositioning(startPoint.X, startPoint.Y);
			if (startPose == nullptr) return;
			PoseSnapshot^ endPose = RunTracIkPositioning(targetPoint.X, targetPoint.Y);
			if (endPose == nullptr) return;

			float maxTheta = static_cast<float>(nudMaxTheta->Value);
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float stretchFactor = static_cast<float>(nudStretchFactor->Value);
			variableLengths = cbVarLen->Checked;
			bool varLen = variableLengths;
			int maxIter = static_cast<int>(nudMaxIter->Value);
			float r_param = static_cast<float>(nudR->Value);
			bool adaptive = cbAdaptive->Checked;
			float eps = currentEps;
			unsigned int seed = static_cast<unsigned int>(GetTickCount());
			int stateDim = nSegments << 1;

			std::vector<float> startState(stateDim);
			for (int i = 0; i < nSegments; ++i)
			{
				startState[i] = startPose->Angles[i];
				startState[nSegments + i] = startPose->Lengths[i];
			}
			std::vector<float> finalState(stateDim);
			for (int i = 0; i < nSegments; ++i)
			{
				finalState[i] = endPose->Angles[i];
				finalState[nSegments + i] = endPose->Lengths[i];
			}

			LARGE_INTEGER startT0, t1, fq;
			QueryPerformanceCounter(&startT0);

			pStart(
				nSegments, varLen, maxTheta, 0.0f, 0.0f, maxIter,
				r_param, adaptive, eps, seed, baseLength, stretchFactor,
				nullptr, 0,
				2, startState.data(), finalState.data()
			);

			float* trajPoints = nullptr;
			int pointCount = 0;
			size_t totalIterations = 0u;

			pBuildTrajectory(
				nSegments, varLen, maxTheta,
				startState.data(), finalState.data(),
				maxIter, r_param, adaptive, eps, seed,
				baseLength, stretchFactor,
				nullptr, 0,
				&trajPoints, &pointCount, &totalIterations
			);

			QueryPerformanceCounter(&t1);
			QueryPerformanceFrequency(&fq);
			float micros = static_cast<float>(1e6 * static_cast<double>(t1.QuadPart - startT0.QuadPart) / static_cast<double>(fq.QuadPart));

			plannedPoses->Clear();
			for (int i = 0; i < pointCount; ++i)
			{
				float* ptr = trajPoints + i * stateDim;
				PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);
				for (int j = 0; j < nSegments; ++j)
				{
					pose->Angles[j] = ptr[j];
					pose->Lengths[j] = varLen ? ptr[nSegments + j] : baseLength;
				}
				ComputeEndEffector(pose->Angles, pose->Lengths, pose->EndX, pose->EndY);
				plannedPoses->Add(pose);
			}
			pFree(trajPoints);

			UpdateTrajectoryStats(plannedPoses, micros, totalIterations, endPose->AchievedEps);
			BuildAnimationFramesFromPlan();
			ApplyPoseToManipulator(plannedPoses[0]);
			StartAnimationIfNeeded();
			this->Invalidate();
		}

		void RunPositioningMode()
		{
			StopAnimation();
			float tx = static_cast<float>(nudTargetX->Value);
			float ty = static_cast<float>(nudTargetY->Value);
			PoseSnapshot^ pose = nullptr;
			if (cbBackend->SelectedIndex == 1)
			{
				pose = RunTracIkPositioning(tx, ty);
			}
			else
			{
				pose = RunAgpCppAtPoint(tx, ty);
			}
			if (pose == nullptr) return;
			ApplyPoseToManipulator(pose);
			UpdatePositioningStats(pose, tx, ty);
			ClearTrajectoryCache();
			this->Invalidate();
			this->Refresh();
		}

		void DrawTrajectorySegmentClipped(Graphics^ g, PointF aWorld, PointF bWorld)
		{
			float prevX = aWorld.X;
			float prevY = aWorld.Y;
			bool prevAllowed = IsDisplayPointAllowed(prevX, prevY);
			for (int s = 1; s <= 20; ++s)
			{
				float t = static_cast<float>(s) / 20.0f;
				float currX = Lerp(aWorld.X, bWorld.X, t);
				float currY = Lerp(aWorld.Y, bWorld.Y, t);
				bool currAllowed = IsDisplayPointAllowed(currX, currY);
				if (prevAllowed && currAllowed)
				{
					PointF pa = WorldToPixel(prevX, prevY);
					PointF pb = WorldToPixel(currX, currY);
					g->DrawLine(pathPen, pa, pb);
				}
				prevX = currX;
				prevY = currY;
				prevAllowed = currAllowed;
			}
		}

		void DrawTrajectoryPath(Graphics^ g)
		{
			if (trajectoryPathWorld->Count < 2) return;
			for (int i = 1; i < trajectoryPathWorld->Count; ++i)
			{
				DrawTrajectorySegmentClipped(g, trajectoryPathWorld[i - 1], trajectoryPathWorld[i]);
			}
			if (plannedPoses->Count > 2)
			{
				for (int i = 1; i < plannedPoses->Count - 1; ++i)
				{
					if (!IsDisplayPointAllowed(plannedPoses[i]->EndX, plannedPoses[i]->EndY)) continue;
					PointF wp = WorldToPixel(plannedPoses[i]->EndX, plannedPoses[i]->EndY);
					g->FillEllipse(waypointBrush, wp.X - 4.0f, wp.Y - 4.0f, 8.0f, 8.0f);
				}
			}
		}

		void RebuildTrajectoryDisplayPathFromPlan()
		{
			trajectoryPathWorld->Clear();
			if (animationFrames->Count > 0)
			{
				for (int i = 0; i < animationFrames->Count; ++i)
				{
					PointF p(animationFrames[i]->EndX, animationFrames[i]->EndY);
					if (trajectoryPathWorld->Count == 0 || DistanceSquared(trajectoryPathWorld[trajectoryPathWorld->Count - 1], p) > 1e-10f)
					{
						trajectoryPathWorld->Add(p);
					}
				}
				return;
			}
			for (int i = 0; i < plannedPoses->Count; ++i)
			{
				PointF p(plannedPoses[i]->EndX, plannedPoses[i]->EndY);
				if (trajectoryPathWorld->Count == 0 || DistanceSquared(trajectoryPathWorld[trajectoryPathWorld->Count - 1], p) > 1e-10f)
				{
					trajectoryPathWorld->Add(p);
				}
			}
		}

		void BuildAnimationFramesFromPlan()
		{
			animationFrames->Clear();
			if (plannedPoses->Count == 0)
			{
				RebuildTrajectoryDisplayPathFromPlan();
				return;
			}
			animationFrames->Add(plannedPoses[0]);
			for (int i = 0; i < plannedPoses->Count - 1; ++i)
			{
				PoseSnapshot^ a = plannedPoses[i];
				PoseSnapshot^ b = plannedPoses[i + 1];
				for (int frame = 1; frame <= AnimationFramesPerSegment; ++frame)
				{
					float t = static_cast<float>(frame) / static_cast<float>(AnimationFramesPerSegment);
					PoseSnapshot^ interp = gcnew PoseSnapshot(nSegments);
					for (int j = 0; j < nSegments; ++j)
					{
						interp->Angles[j] = LerpWrappedAngle(a->Angles[j], b->Angles[j], t);
						interp->Lengths[j] = variableLengths
							? a->Lengths[j] + t * (b->Lengths[j] - a->Lengths[j])
							: static_cast<float>(nudBaseLength->Value);
					}
					ComputeEndEffector(interp->Angles, interp->Lengths, interp->EndX, interp->EndY);
					animationFrames->Add(interp);
				}
			}
			RebuildTrajectoryDisplayPathFromPlan();
		}

		void StartAnimationIfNeeded()
		{
			if (animationFrames->Count <= 1)
			{
				animationRunning = false;
				this->Invalidate();
				return;
			}
			animationFrameIndex = 0;
			animationRunning = true;
			animationTimer->Start();
		}

		System::Void OnAnimationTick(System::Object^ sender, System::EventArgs^ e)
		{
			if (!animationRunning || animationFrames->Count == 0)
			{
				StopAnimation();
				return;
			}
			if (animationFrameIndex >= animationFrames->Count)
			{
				StopAnimation();
				return;
			}
			ApplyPoseToManipulator(animationFrames[animationFrameIndex]);
			++animationFrameIndex;
			this->Invalidate();
			if (animationFrameIndex >= animationFrames->Count) StopAnimation();
		}

		PointF WorldToPixel(float wx, float wy)
		{
			PointF basePoint = GetBasePoint();
			return PointF(basePoint.X + wx * 160.0f, basePoint.Y - wy * 160.0f);
		}

		float GetCurrentMaxReach()
		{
			float baseLength = static_cast<float>(nudBaseLength->Value);
			float stretchFactor = static_cast<float>(nudStretchFactor->Value);
			return static_cast<float>(nSegments) * (cbVarLen->Checked ? baseLength * stretchFactor : baseLength);
		}

		bool IsDisplayPointAllowed(float x, float y)
		{
			float maxReach = GetCurrentMaxReach();
			if (x * x + y * y > (maxReach + 0.05f) * (maxReach + 0.05f)) return false;

			if (obstacleX->Count > 0)
			{
				for (int i = 0; i < obstacleX->Count; ++i)
				{
					float half = obstacleHalf[i] + ObstacleClearance;
					if (x >= obstacleX[i] - half && x <= obstacleX[i] + half &&
						y >= obstacleY[i] - half && y <= obstacleY[i] + half) return false;
				}
			}
			return true;
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		System::Void OnMouseDownPoint(System::Object^ sender, MouseEventArgs^ e)
		{
			PointF basePoint = GetBasePoint();
			float baseX = basePoint.X;
			float baseY = basePoint.Y;
			float targetX = static_cast<float>(nudTargetX->Value);
			float targetY = static_cast<float>(nudTargetY->Value);
			float pixelTargetX = baseX + targetX * 160.0f;
			float pixelTargetY = baseY - targetY * 160.0f;

			float bestDistSquared = 100.0f;
			activeDragHandle = DragHandle::None;

			if (IsTrajectoryMode())
			{
				float startX = static_cast<float>(nudStartX->Value);
				float startY = static_cast<float>(nudStartY->Value);
				float pixelStartX = baseX + startX * 160.0f;
				float pixelStartY = baseY - startY * 160.0f;
				float dsx = e->X - pixelStartX;
				float dsy = e->Y - pixelStartY;
				float startDistSquared = dsx * dsx + dsy * dsy;
				if (startDistSquared < bestDistSquared)
				{
					bestDistSquared = startDistSquared;
					activeDragHandle = DragHandle::Start;
				}
			}

			float dtx = e->X - pixelTargetX;
			float dty = e->Y - pixelTargetY;
			float targetDistSquared = dtx * dtx + dty * dty;
			if (targetDistSquared < bestDistSquared) activeDragHandle = DragHandle::Target;

			if (activeDragHandle != DragHandle::None)
			{
				StopAnimation();
				updatingFromMouse = true;
				this->Capture = true;
			}
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		System::Void OnMouseMovePoint(System::Object^ sender, MouseEventArgs^ e)
		{
			if (activeDragHandle == DragHandle::None) return;

			PointF basePoint = GetBasePoint();
			float baseX = basePoint.X;
			float baseY = basePoint.Y;
			float minX = static_cast<float>(nudTargetX->Minimum);
			float maxX = static_cast<float>(nudTargetX->Maximum);
			float minY = static_cast<float>(nudTargetY->Minimum);
			float maxY = static_cast<float>(nudTargetY->Maximum);

			float newX = (e->X - baseX) / 160.0f;
			float newY = (baseY - e->Y) / 160.0f;
			newX = Math::Max(minX, Math::Min(maxX, newX));
			newY = Math::Max(minY, Math::Min(maxY, newY));

			if (activeDragHandle == DragHandle::Target)
			{
				nudTargetX->Value = Decimal(newX);
				nudTargetY->Value = Decimal(newY);
			}
			else
			{
				startPointCustomized = true;
				nudStartX->Value = Decimal(newX);
				nudStartY->Value = Decimal(newY);
			}

			ClearTrajectoryCache();
			this->Invalidate();
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		System::Void OnMouseUpPoint(System::Object^ sender, MouseEventArgs^ e)
		{
			if (activeDragHandle != DragHandle::None)
			{
				activeDragHandle = DragHandle::None;
				updatingFromMouse = false;
				this->Capture = false;
			}
		}

		System::Void OnBackendChanged(System::Object^ sender, System::EventArgs^ e)
		{
			UpdateBackendUiState();
		}

		System::Void OnDemoModeChanged(System::Object^ sender, System::EventArgs^ e)
		{
			demoMode = (cbDemoMode->SelectedIndex == 0) ? DemoMode::Positioning : DemoMode::TrajectoryPlanning;
			if (demoMode == DemoMode::TrajectoryPlanning && !startPointCustomized) SetDefaultStartPoint(false);
			UpdateTrajectoryUiState();
		}

		System::Void OnResize(System::Object^ sender, System::EventArgs^ e)
		{
			this->Invalidate();
		}

		System::Void OnAnyChanged(System::Object^ sender, System::EventArgs^ e)
		{
			if (updatingFromMouse) return;
			ClearTrajectoryCache();
			this->Invalidate();
		}

		System::Void OnTargetChanged(System::Object^ sender, System::EventArgs^ e)
		{
			if (updatingFromMouse) return;
			if (!startPointCustomized) SetDefaultStartPoint(false);
			ClearTrajectoryCache();
			this->Invalidate();
		}

		System::Void OnStartPointChanged(System::Object^ sender, System::EventArgs^ e)
		{
			if (updatingFromMouse) return;
			if (!syncingDefaultStartPoint) startPointCustomized = true;
			ClearTrajectoryCache();
			this->Invalidate();
		}

		System::Void OnAddClick(System::Object^ sender, System::EventArgs^ e)
		{
			++nSegments;
			angles->Add(0.0f);
			lengths->Add(static_cast<float>(nudBaseLength->Value));
			ClearTrajectoryCache();
			this->Invalidate();
		}

		System::Void OnRemClick(System::Object^ sender, System::EventArgs^ e)
		{
			if (nSegments > 1)
			{
				--nSegments;
				angles->RemoveAt(angles->Count - 1);
				lengths->RemoveAt(lengths->Count - 1);
				ClearTrajectoryCache();
				this->Invalidate();
			}
		}

		System::Void OnOptimizeClick(System::Object^ sender, System::EventArgs^ e)
		{
			if (IsTrajectoryMode())
			{
				RunTrajectoryPlanningMode();
			}
			else
			{
				RunPositioningMode();
			}
		}

	protected:
		[MethodImpl(MethodImplOptions::AggressiveInlining)]
			virtual void OnPaint(PaintEventArgs^ e) override
		{
			Form::OnPaint(e);

			Graphics^ g = e->Graphics;
			g->SmoothingMode = SmoothingMode::HighQuality;
			g->Clear(this->BackColor);

			System::Drawing::Rectangle drawArea(0, 180, this->ClientSize.Width, this->ClientSize.Height - 180);
			g->FillRectangle(Brushes::White, drawArea);

			float targetX = static_cast<float>(nudTargetX->Value);
			float targetY = static_cast<float>(nudTargetY->Value);
			PointF basePoint = GetBasePoint();
			int baseX = static_cast<int>(basePoint.X);
			int baseY = static_cast<int>(basePoint.Y);
			float pixelTargetX = baseX + targetX * 160.0f;
			float pixelTargetY = baseY - targetY * 160.0f;

			g->DrawLine(wallPen, baseX - 25, baseY + 8, baseX + 25, baseY + 8);
			g->FillRectangle(wallHatchBrush, baseX - 25, baseY + 8, 50, 12);

			g->DrawLine(dashedPen, pixelTargetX - 25, pixelTargetY + 8, pixelTargetX + 25, pixelTargetY + 8);
			g->FillRectangle(wallHatchBrush, static_cast<int>(pixelTargetX) - 25, static_cast<int>(pixelTargetY) + 8, 50, 12);
			g->DrawEllipse(targetPen, pixelTargetX - 8.0f, pixelTargetY - 8.0f, 16.0f, 16.0f);

			if (IsTrajectoryMode())
			{
				float startX = static_cast<float>(nudStartX->Value);
				float startY = static_cast<float>(nudStartY->Value);
				float pixelStartX = baseX + startX * 160.0f;
				float pixelStartY = baseY - startY * 160.0f;
				g->DrawLine(dashedPen, pixelStartX - 25, pixelStartY + 8, pixelStartX + 25, pixelStartY + 8);
				g->FillRectangle(wallHatchBrush, static_cast<int>(pixelStartX) - 25, static_cast<int>(pixelStartY) + 8, 50, 12);
				g->DrawEllipse(startPen, pixelStartX - 8.0f, pixelStartY - 8.0f, 16.0f, 16.0f);
			}

			if (trajectoryPathWorld->Count > 1 && IsTrajectoryMode()) DrawTrajectoryPath(g);

			if (obstacleX->Count > 0)
			{
				array<float>^ xs = obstacleX->ToArray();
				array<float>^ ys = obstacleY->ToArray();
				array<float>^ hs = obstacleHalf->ToArray();
				for (int obsIdx = 0; obsIdx < xs->Length; ++obsIdx)
				{
					float cx = xs[obsIdx];
					float cy = ys[obsIdx];
					float half = hs[obsIdx];
					float left = static_cast<float>(baseX) + (cx - half) * 160.0f;
					float top = static_cast<float>(baseY) - (cy + half) * 160.0f;
					float size = 2.0f * half * 160.0f;
					float marginLeft = static_cast<float>(baseX) + (cx - half - ObstacleClearance) * 160.0f;
					float marginTop = static_cast<float>(baseY) - (cy + half + ObstacleClearance) * 160.0f;
					float marginSize = 2.0f * (half + ObstacleClearance) * 160.0f;
					g->DrawRectangle(obstacleMarginPen, marginLeft, marginTop, marginSize, marginSize);
					g->FillRectangle(obstacleBrush, left, top, size, size);
					g->DrawRectangle(obstaclePen, left, top, size, size);
				}
			}

			cli::array<PointF>^ pts = gcnew cli::array<PointF>(nSegments + 1);
			pts[0] = PointF(static_cast<float>(baseX), static_cast<float>(baseY));

			float x = 0.0f;
			float y = 0.0f;
			float phi = PI_2;
			array<float>^ localAngles = angles->ToArray();
			array<float>^ localLengths = lengths->ToArray();

			for (int i = 0; i < nSegments; ++i)
			{
				float theta = localAngles[i];
				float L = localLengths[i];
				phi += theta;
				x += L * static_cast<float>(Math::Cos(static_cast<double>(phi)));
				y += L * static_cast<float>(Math::Sin(static_cast<double>(phi)));
				pts[i + 1] = PointF(static_cast<float>(baseX) + x * 160.0f, static_cast<float>(baseY) - y * 160.0f);
			}

			for (int i = 0; i < nSegments; ++i) g->DrawLine(penRod, pts[i], pts[i + 1]);
			for (int i = 0; i <= nSegments; ++i) g->FillEllipse(jointBrush, pts[i].X - 8.0f, pts[i].Y - 8.0f, 16.0f, 16.0f);
		}
	};
}
