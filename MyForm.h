#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cmath>
#include <vector>
#include <memory>

#include <kdl/chain.hpp>
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
using namespace System::Runtime::InteropServices;

typedef void(__cdecl* P_MANIP)(int, bool, float, float, float, float, int, float, bool, float, unsigned int, float, float, const float*, int, float**, float*, float*, float*, float*, size_t*, float*, int, const float*);
typedef void(__cdecl* P_START)(int, bool, float, float, float, float, int, float, bool, float, unsigned int, float, float, const float*, int, int, const float*, const float*);
typedef void(__cdecl* P_BUILD_TRAJECTORY)(int, bool, float, const float*, const float*, int, float, bool, float, unsigned int, float, float, const float*, int, float**, int*, size_t*);

class TracIkRunner final
{
private:
	int nSegments_;
	unsigned int nJoints_;
	double baseLength_;
	double maxTheta_;
	double maxTime_;
	double eps_;

	KDL::Chain chain_;
	KDL::JntArray qMin_;
	KDL::JntArray qMax_;
	KDL::JntArray qInit_;
	KDL::JntArray qOut_;

	std::unique_ptr<trac_ik::TRAC_IK> solver_;
	LARGE_INTEGER frequency_;

	static double WrapPi(double a)
	{
		const double pi = 3.1415926535897932384626433832795;
		const double twoPi = 2.0 * pi;

		while (a > pi)
			a -= twoPi;

		while (a < -pi)
			a += twoPi;

		return a;
	}

public:
	TracIkRunner(
		int nSegments,
		double baseLength,
		double maxTheta,
		double maxTime,
		double eps)
		:
		nSegments_(nSegments),
		nJoints_(0),
		baseLength_(baseLength),
		maxTheta_(maxTheta),
		maxTime_(maxTime),
		eps_(eps)
	{
		for (int i = 0; i < nSegments_; ++i)
		{
			chain_.addSegment(
				KDL::Segment(
					KDL::Joint(KDL::Joint::RotZ),
					KDL::Frame(
						KDL::Vector(
							baseLength_,
							0.0,
							0.0
						)
					)
				)
			);
		}

		nJoints_ = chain_.getNrOfJoints();

		qMin_.resize(nJoints_);
		qMax_.resize(nJoints_);
		qInit_.resize(nJoints_);
		qOut_.resize(nJoints_);

		for (unsigned int i = 0; i < nJoints_; ++i)
		{
			qMin_(i) = -maxTheta_;
			qMax_(i) = maxTheta_;
			qInit_(i) = 0.0;
			qOut_(i) = 0.0;
		}

		solver_ = std::make_unique<trac_ik::TRAC_IK>(
			chain_,
			qMin_,
			qMax_,
			maxTime_,
			eps_,
			trac_ik::SolveType::Speed
		);

		QueryPerformanceFrequency(&frequency_);
	}

	TracIkRunner(const TracIkRunner&) = delete;
	TracIkRunner& operator=(const TracIkRunner&) = delete;

	bool Matches(
		int nSegments,
		double baseLength,
		double maxTheta,
		double maxTime,
		double eps) const
	{
		return
			nSegments_ == nSegments &&
			baseLength_ == baseLength &&
			maxTheta_ == maxTheta &&
			maxTime_ == maxTime &&
			eps_ == eps;
	}

	bool Solve(
		float targetX,
		float targetY,
		float targetAngle,
		double& outBestF,
		double& outBestX,
		double& outBestY,
		double& outBestA,
		double& outMillis)
	{
		for (unsigned int i = 0; i < nJoints_; ++i)
		{
			qInit_(i) = 0.0;
			qOut_(i) = 0.0;
		}

		KDL::Frame target(
			KDL::Rotation::RotZ(
				static_cast<double>(targetAngle)
			),
			KDL::Vector(
				static_cast<double>(targetX),
				static_cast<double>(targetY),
				0.0
			)
		);

		KDL::Twist tolerances(
			KDL::Vector(
				eps_,
				eps_,
				0.0
			),
			KDL::Vector(
				0.0,
				0.0,
				eps_
			)
		);

		LARGE_INTEGER t0;
		LARGE_INTEGER t1;

		QueryPerformanceCounter(&t0);

		const int result = solver_->CartToJnt(
			qInit_,
			target,
			qOut_,
			tolerances
		);

		QueryPerformanceCounter(&t1);

		outMillis =
			1.0e3 *
			static_cast<double>(
				t1.QuadPart - t0.QuadPart
				) /
			static_cast<double>(
				frequency_.QuadPart
				);

		double x = 0.0;
		double y = 0.0;
		double phi = 0.0;

		for (unsigned int i = 0; i < nJoints_; ++i)
		{
			phi += qOut_(i);
			x += baseLength_ * std::cos(phi);
			y += baseLength_ * std::sin(phi);
		}

		const double dx =
			x - static_cast<double>(targetX);

		const double dy =
			y - static_cast<double>(targetY);

		const double da =
			WrapPi(
				phi - static_cast<double>(targetAngle)
			);

		outBestF =
			std::sqrt(
				dx * dx +
				dy * dy +
				da * da
			);

		outBestX = x;
		outBestY = y;
		outBestA = WrapPi(phi);

		return result >= 0;
	}

	const KDL::JntArray& GetSolution() const
	{
		return qOut_;
	}
};

namespace TESTAGP
{
	public enum class DemoMode { Positioning = 0, TrajectoryPlanning = 1 };
	public enum class DragHandle { None = 0, Target = 1, Start = 2, AngleTarget = 3, AngleStart = 4 };

	public ref class PoseSnapshot sealed
	{
	public:
		PoseSnapshot(int n)
		{
			Angles = gcnew array<float>(n);
			Lengths = gcnew array<float>(n);
			EndX = EndY = EndA = BestF = 0.0f;
			Iterations = 0;
			AchievedEps = Millis = 0.0f;
		}

		array<float>^ Angles;
		array<float>^ Lengths;
		float EndX, EndY, EndA, BestF;
		int Iterations;
		float AchievedEps, Millis;
	};

	public ref class MyForm sealed : public Form
	{
	public:
		MyForm(HMODULE hLib)
			:
			hLib(hLib),
			tracIkRunner(nullptr)
		{
			this->SetStyle(ControlStyles::AllPaintingInWmPaint | ControlStyles::UserPaint | ControlStyles::OptimizedDoubleBuffer, true);
			this->Text = L"AGP Manipulator 2D";
			this->ClientSize = System::Drawing::Size(1200, 800);
			this->Resize += gcnew EventHandler(this, &MyForm::OnResize);
			this->MouseDown += gcnew MouseEventHandler(this, &MyForm::OnMouseDownPoint);
			this->MouseMove += gcnew MouseEventHandler(this, &MyForm::OnMouseMovePoint);
			this->MouseUp += gcnew MouseEventHandler(this, &MyForm::OnMouseUpPoint);

			fManip = (P_MANIP)GetProcAddress(hLib, "AGP_Manip2D");
			pStart = (P_START)GetProcAddress(hLib, "AgpStartManipND");
			pBuildTrajectory = (P_BUILD_TRAJECTORY)GetProcAddress(hLib, "AGP_BuildTransitionTrajectory");

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
			InitAnimation();
			InitUI();
			ResetRandomConfig();
			SetDefaultStartPoint(true);
			UpdateTrajectoryUiState();
		}

		~MyForm()
		{
			this->!MyForm();
		}

		!MyForm()
		{
			if (tracIkRunner != nullptr)
			{
				delete tracIkRunner;
				tracIkRunner = nullptr;
			}
		}

	private:
		literal float ObstacleClearance = 0.05f;
		literal float PI = 3.14159265358979323846f;
		literal float TWO_PI = 6.28318530717958647692f;
		literal int AnimationFramesPerSegment = 10;
		literal int AnimationIntervalMs = 16;
		literal float TransitionLengthEnergyWeight = 0.35f;
		literal float TransitionPrefixEnergyWeight = 0.175f;

		ComboBox^ cbDemoMode;
		ComboBox^ cbBackend;
		CheckBox^ cbVarLen;
		CheckBox^ cbAdaptive;
		NumericUpDown^ nudMaxTheta;
		NumericUpDown^ nudBaseLength;
		NumericUpDown^ nudStretchFactor;
		NumericUpDown^ nudTargetX;
		NumericUpDown^ nudTargetY;
		NumericUpDown^ nudTargetAngle;
		NumericUpDown^ nudStartX;
		NumericUpDown^ nudStartY;
		NumericUpDown^ nudStartAngle;
		NumericUpDown^ nudMaxIter;
		NumericUpDown^ nudR;
		TextBox^ txtEps;
		float currentEps = 1e-9f;
		Button^ btnAdd;
		Button^ btnRem;
		Button^ btnOptimize;
		Button^ btnGenerateObstacles;
		Button^ btnClearObstacles;
		Label^ lblInfo;
		Label^ lblStartX;
		Label^ lblStartY;
		Label^ lblTargetAngle;
		Label^ lblStartAngle;

		P_BUILD_TRAJECTORY pBuildTrajectory;
		HMODULE hLib;
		P_MANIP fManip;
		P_START pStart;

		TracIkRunner* tracIkRunner;

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
		Pen^ angleTargetPen;
		Pen^ angleStartPen;

		System::Drawing::Font^ uiFontBold11;
		System::Drawing::Font^ uiFontTextBox;
		System::Drawing::Font^ uiFontBold10;

		Timer^ animationTimer;

		DemoMode demoMode;
		int nSegments = 1;
		bool variableLengths = false;
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
		bool angleTargetDrag = false;
		bool angleStartDrag = false;

		static float WrapPi(float a)
		{
			while (a > PI)
				a -= TWO_PI;

			while (a < -PI)
				a += TWO_PI;

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
			int drawAreaTop = 180;
			int drawAreaHeight = this->ClientSize.Height - 180;
			int leftWallX = this->ClientSize.Width * 25 / 100;
			return PointF((float)leftWallX, (float)(drawAreaTop + drawAreaHeight / 2));
		}

		TracIkRunner* GetTracIkRunner()
		{
			const double maxTime = 0.002;
			const double baseLengthValue = static_cast<double>((float)nudBaseLength->Value);
			const double maxThetaValue = static_cast<double>((float)nudMaxTheta->Value);
			const double epsValue = static_cast<double>(currentEps);

			if (tracIkRunner == nullptr ||
				!tracIkRunner->Matches(
					nSegments,
					baseLengthValue,
					maxThetaValue,
					maxTime,
					epsValue))
			{
				if (tracIkRunner != nullptr)
				{
					delete tracIkRunner;
					tracIkRunner = nullptr;
				}

				tracIkRunner = new TracIkRunner(
					nSegments,
					baseLengthValue,
					maxThetaValue,
					maxTime,
					epsValue
				);
			}

			return tracIkRunner;
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
			angleTargetPen = gcnew Pen(Color::Red, 2.0f);
			angleTargetPen->DashStyle = DashStyle::Dash;
			angleStartPen = gcnew Pen(Color::DarkOrange, 2.0f);
			angleStartPen->DashStyle = DashStyle::Dash;
		}

		void InitAnimation()
		{
			animationTimer = gcnew Timer();
			animationTimer->Interval = AnimationIntervalMs;
			animationTimer->Tick += gcnew EventHandler(this, &MyForm::OnAnimationTick);
		}

		void InitUI()
		{
			cbDemoMode = gcnew ComboBox(); cbDemoMode->Location = Point(920, 20); cbDemoMode->Width = 260; cbDemoMode->Height = 28; cbDemoMode->DropDownStyle = ComboBoxStyle::DropDownList; cbDemoMode->Font = uiFontBold11; cbDemoMode->BackColor = SystemColors::Info; cbDemoMode->FlatStyle = FlatStyle::Flat; cbDemoMode->Items->Add(L"Позиционирование"); cbDemoMode->Items->Add(L"Планирование траектории"); cbDemoMode->SelectedIndex = 0; cbDemoMode->SelectedIndexChanged += gcnew EventHandler(this, &MyForm::OnDemoModeChanged); this->Controls->Add(cbDemoMode);
			cbBackend = gcnew ComboBox(); cbBackend->Location = Point(920, 54); cbBackend->Width = 260; cbBackend->Height = 28; cbBackend->DropDownStyle = ComboBoxStyle::DropDownList; cbBackend->Font = uiFontBold11; cbBackend->BackColor = SystemColors::Info; cbBackend->FlatStyle = FlatStyle::Flat; cbBackend->Items->Add(L"AGP"); cbBackend->Items->Add(L"TRAC-IK"); cbBackend->SelectedIndex = 0; cbBackend->SelectedIndexChanged += gcnew EventHandler(this, &MyForm::OnBackendChanged); this->Controls->Add(cbBackend);
			Label^ L = gcnew Label(); L->Text = L"Макс. угол (рад.)"; L->Location = Point(20, 20); L->Width = 200; L->Font = uiFontBold11; this->Controls->Add(L);
			nudMaxTheta = gcnew NumericUpDown(); nudMaxTheta->Location = Point(20, 52); nudMaxTheta->Width = 200; nudMaxTheta->DecimalPlaces = 3; nudMaxTheta->Minimum = Decimal(0.01); nudMaxTheta->Maximum = Decimal(3.14159); nudMaxTheta->Value = Decimal(2.0); nudMaxTheta->Font = uiFontTextBox; nudMaxTheta->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(nudMaxTheta);
			L = gcnew Label(); L->Text = L"Базовая длина"; L->Location = Point(245, 20); L->Width = 200; L->Font = uiFontBold11; this->Controls->Add(L);
			nudBaseLength = gcnew NumericUpDown(); nudBaseLength->Location = Point(245, 52); nudBaseLength->Width = 200; nudBaseLength->DecimalPlaces = 2; nudBaseLength->Minimum = Decimal(0.5); nudBaseLength->Maximum = Decimal(2.0); nudBaseLength->Value = Decimal(1.0); nudBaseLength->Font = uiFontTextBox; nudBaseLength->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(nudBaseLength);
			L = gcnew Label(); L->Text = L"Макс. коэфф. растяжения/сжатия"; L->Location = Point(470, 20); L->Width = 300; L->Font = uiFontBold11; this->Controls->Add(L);
			nudStretchFactor = gcnew NumericUpDown(); nudStretchFactor->Location = Point(470, 52); nudStretchFactor->Width = 200; nudStretchFactor->DecimalPlaces = 2; nudStretchFactor->Minimum = Decimal(1.0); nudStretchFactor->Maximum = Decimal(1.5); nudStretchFactor->Increment = Decimal(0.01); nudStretchFactor->Value = Decimal(1.5); nudStretchFactor->Font = uiFontTextBox; nudStretchFactor->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(nudStretchFactor);
			cbVarLen = gcnew CheckBox(); cbVarLen->Text = L"Переменные длины"; cbVarLen->Location = Point(695, 52); cbVarLen->Width = 200; cbVarLen->Checked = false; cbVarLen->Font = uiFontBold11; cbVarLen->CheckedChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(cbVarLen);
			L = gcnew Label(); L->Text = L"Цель X"; L->Location = Point(20, 107); L->Width = 200; L->Font = uiFontBold11; this->Controls->Add(L);
			nudTargetX = gcnew NumericUpDown(); nudTargetX->Location = Point(20, 139); nudTargetX->Width = 200; nudTargetX->DecimalPlaces = 2; nudTargetX->Minimum = Decimal(-10.0); nudTargetX->Maximum = Decimal(10.0); nudTargetX->Value = Decimal(2.5); nudTargetX->Font = uiFontTextBox; nudTargetX->ValueChanged += gcnew EventHandler(this, &MyForm::OnTargetChanged); this->Controls->Add(nudTargetX);
			L = gcnew Label(); L->Text = L"Цель Y"; L->Location = Point(245, 107); L->Width = 200; L->Font = uiFontBold11; this->Controls->Add(L);
			nudTargetY = gcnew NumericUpDown(); nudTargetY->Location = Point(245, 139); nudTargetY->Width = 200; nudTargetY->DecimalPlaces = 2; nudTargetY->Minimum = Decimal(-10.0); nudTargetY->Maximum = Decimal(10.0); nudTargetY->Value = Decimal(-1.0); nudTargetY->Font = uiFontTextBox; nudTargetY->ValueChanged += gcnew EventHandler(this, &MyForm::OnTargetChanged); this->Controls->Add(nudTargetY);
			lblTargetAngle = gcnew Label(); lblTargetAngle->Text = L"Цель угол (рад.)"; lblTargetAngle->Location = Point(20, 194); lblTargetAngle->Width = 200; lblTargetAngle->Font = uiFontBold11; this->Controls->Add(lblTargetAngle);
			nudTargetAngle = gcnew NumericUpDown(); nudTargetAngle->Location = Point(20, 226); nudTargetAngle->Width = 200; nudTargetAngle->DecimalPlaces = 3; nudTargetAngle->Minimum = Decimal(-1000); nudTargetAngle->Maximum = Decimal(1000); nudTargetAngle->Increment = Decimal(0.1); nudTargetAngle->Value = Decimal(0); nudTargetAngle->Font = uiFontTextBox; nudTargetAngle->ValueChanged += gcnew EventHandler(this, &MyForm::OnAngleChanged); this->Controls->Add(nudTargetAngle);
			L = gcnew Label(); L->Text = L"Надежность (r)"; L->Location = Point(470, 107); L->Width = 200; L->Font = uiFontBold11; this->Controls->Add(L);
			nudR = gcnew NumericUpDown(); nudR->Location = Point(470, 139); nudR->Width = 200; nudR->DecimalPlaces = 2; nudR->Minimum = Decimal(1.0); nudR->Maximum = Decimal(20.0); nudR->Value = Decimal(1.05); nudR->Font = uiFontTextBox; nudR->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(nudR);
			L = gcnew Label(); L->Text = L"Точность"; L->Location = Point(695, 107); L->Width = 100; L->Font = uiFontBold11; this->Controls->Add(L);
			txtEps = gcnew TextBox(); txtEps->Location = Point(695, 139); txtEps->Width = 80; txtEps->Font = uiFontTextBox; txtEps->Text = L"1E-09"; txtEps->TextChanged += gcnew EventHandler(this, &MyForm::OnEpsTextChanged); this->Controls->Add(txtEps);
			Button^ btnEpsUp = gcnew Button(); btnEpsUp->Text = L"×10"; btnEpsUp->Location = Point(780, 139); btnEpsUp->Width = 32; btnEpsUp->Height = 26; btnEpsUp->Font = uiFontBold11; btnEpsUp->TextAlign = ContentAlignment::TopRight; btnEpsUp->Padding = System::Windows::Forms::Padding(0, 0, 2, 3); btnEpsUp->Click += gcnew EventHandler(this, &MyForm::OnEpsOrderUp); this->Controls->Add(btnEpsUp);
			Button^ btnEpsDown = gcnew Button(); btnEpsDown->Text = L"÷10"; btnEpsDown->Location = Point(816, 139); btnEpsDown->Width = 32; btnEpsDown->Height = 26; btnEpsDown->Font = uiFontBold11; btnEpsDown->TextAlign = ContentAlignment::TopRight; btnEpsDown->Padding = System::Windows::Forms::Padding(0, 0, 2, 3); btnEpsDown->Click += gcnew EventHandler(this, &MyForm::OnEpsOrderDown); this->Controls->Add(btnEpsDown);
			L = gcnew Label(); L->Text = L"Макс. итераций"; L->Location = Point(860, 107); L->Width = 130; L->Font = uiFontBold11; this->Controls->Add(L);
			nudMaxIter = gcnew NumericUpDown(); nudMaxIter->Location = Point(860, 139); nudMaxIter->Width = 120; nudMaxIter->Minimum = 10; nudMaxIter->Maximum = 500000; nudMaxIter->Value = 1000; nudMaxIter->Font = uiFontTextBox; nudMaxIter->Increment = 100; nudMaxIter->ValueChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(nudMaxIter);
			cbAdaptive = gcnew CheckBox(); cbAdaptive->Text = L"Адаптивная схема"; cbAdaptive->Location = Point(995, 139); cbAdaptive->Width = 150; cbAdaptive->Checked = true; cbAdaptive->Font = uiFontBold11; cbAdaptive->CheckedChanged += gcnew EventHandler(this, &MyForm::OnAnyChanged); this->Controls->Add(cbAdaptive);
			btnAdd = gcnew Button(); btnAdd->Text = L"+ Звено"; btnAdd->Location = Point(465, 211); btnAdd->Width = 90; btnAdd->Height = 35; btnAdd->BackColor = SystemColors::Info; btnAdd->Cursor = Cursors::Hand; btnAdd->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64); btnAdd->FlatAppearance->BorderSize = 3; btnAdd->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255); btnAdd->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255); btnAdd->FlatStyle = FlatStyle::Flat; btnAdd->Font = uiFontBold11; btnAdd->ForeColor = SystemColors::ControlDarkDark; btnAdd->Click += gcnew EventHandler(this, &MyForm::OnAddClick); this->Controls->Add(btnAdd);
			btnRem = gcnew Button(); btnRem->Text = L"- Звено"; btnRem->Location = Point(560, 211); btnRem->Width = 90; btnRem->Height = 35; btnRem->BackColor = SystemColors::Info; btnRem->Cursor = Cursors::Hand; btnRem->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64); btnRem->FlatAppearance->BorderSize = 3; btnRem->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255); btnRem->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255); btnRem->FlatStyle = FlatStyle::Flat; btnRem->Font = uiFontBold11; btnRem->ForeColor = SystemColors::ControlDarkDark; btnRem->Click += gcnew EventHandler(this, &MyForm::OnRemClick); this->Controls->Add(btnRem);
			btnOptimize = gcnew Button(); btnOptimize->Text = L"Оптимизировать"; btnOptimize->Location = Point(680, 211); btnOptimize->Width = 150; btnOptimize->Height = 35; btnOptimize->BackColor = SystemColors::Info; btnOptimize->Cursor = Cursors::Hand; btnOptimize->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64); btnOptimize->FlatAppearance->BorderSize = 3; btnOptimize->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255); btnOptimize->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255); btnOptimize->FlatStyle = FlatStyle::Flat; btnOptimize->Font = uiFontBold11; btnOptimize->ForeColor = SystemColors::ControlDarkDark; btnOptimize->Click += gcnew EventHandler(this, &MyForm::OnOptimizeClick); this->Controls->Add(btnOptimize);
			btnGenerateObstacles = gcnew Button(); btnGenerateObstacles->Location = Point(465, 257); btnGenerateObstacles->Width = 365; btnGenerateObstacles->Height = 35; btnGenerateObstacles->BackColor = SystemColors::Info; btnGenerateObstacles->Cursor = Cursors::Hand; btnGenerateObstacles->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64); btnGenerateObstacles->FlatAppearance->BorderSize = 3; btnGenerateObstacles->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255); btnGenerateObstacles->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255); btnGenerateObstacles->FlatStyle = FlatStyle::Flat; btnGenerateObstacles->Font = uiFontBold11; btnGenerateObstacles->ForeColor = SystemColors::ControlDarkDark; btnGenerateObstacles->Click += gcnew EventHandler(this, &MyForm::OnGenerateObstaclesClick); this->Controls->Add(btnGenerateObstacles);
			btnClearObstacles = gcnew Button(); btnClearObstacles->Location = Point(465, 304); btnClearObstacles->Width = 365; btnClearObstacles->Height = 35; btnClearObstacles->BackColor = SystemColors::Info; btnClearObstacles->Cursor = Cursors::Hand; btnClearObstacles->FlatAppearance->BorderColor = Color::FromArgb(64, 64, 64); btnClearObstacles->FlatAppearance->BorderSize = 3; btnClearObstacles->FlatAppearance->MouseDownBackColor = Color::FromArgb(128, 128, 255); btnClearObstacles->FlatAppearance->MouseOverBackColor = Color::FromArgb(192, 192, 255); btnClearObstacles->FlatStyle = FlatStyle::Flat; btnClearObstacles->Font = uiFontBold11; btnClearObstacles->ForeColor = SystemColors::ControlDarkDark; btnClearObstacles->Text = L"Очистить"; btnClearObstacles->Click += gcnew EventHandler(this, &MyForm::OnClearObstaclesClick); this->Controls->Add(btnClearObstacles);
			lblInfo = gcnew Label(); lblInfo->Location = Point(835, 194); lblInfo->Size = System::Drawing::Size(275, 145); lblInfo->BorderStyle = BorderStyle::FixedSingle; lblInfo->Font = uiFontBold10; this->Controls->Add(lblInfo);
			lblStartX = gcnew Label(); lblStartX->Text = L"Начало X"; lblStartX->Location = Point(20, 272); lblStartX->Width = 200; lblStartX->Font = uiFontBold11; this->Controls->Add(lblStartX);
			nudStartX = gcnew NumericUpDown(); nudStartX->Location = Point(20, 304); nudStartX->Width = 200; nudStartX->DecimalPlaces = 2; nudStartX->Minimum = Decimal(-10.0); nudStartX->Maximum = Decimal(10.0); nudStartX->Value = Decimal(1.25); nudStartX->Font = uiFontTextBox; nudStartX->ValueChanged += gcnew EventHandler(this, &MyForm::OnStartPointChanged); this->Controls->Add(nudStartX);
			lblStartY = gcnew Label(); lblStartY->Text = L"Начало Y"; lblStartY->Location = Point(245, 272); lblStartY->Width = 200; lblStartY->Font = uiFontBold11; this->Controls->Add(lblStartY);
			nudStartY = gcnew NumericUpDown(); nudStartY->Location = Point(245, 304); nudStartY->Width = 200; nudStartY->DecimalPlaces = 2; nudStartY->Minimum = Decimal(-10.0); nudStartY->Maximum = Decimal(10.0); nudStartY->Value = Decimal(-0.5); nudStartY->Font = uiFontTextBox; nudStartY->ValueChanged += gcnew EventHandler(this, &MyForm::OnStartPointChanged); this->Controls->Add(nudStartY);
			lblStartAngle = gcnew Label(); lblStartAngle->Text = L"Нач. угол (рад.)"; lblStartAngle->Location = Point(20, 340); lblStartAngle->Width = 200; lblStartAngle->Font = uiFontBold11; this->Controls->Add(lblStartAngle);
			nudStartAngle = gcnew NumericUpDown(); nudStartAngle->Location = Point(20, 372); nudStartAngle->Width = 200; nudStartAngle->DecimalPlaces = 3; nudStartAngle->Minimum = Decimal(-1000); nudStartAngle->Maximum = Decimal(1000); nudStartAngle->Increment = Decimal(0.1); nudStartAngle->Value = Decimal(0); nudStartAngle->Font = uiFontTextBox; nudStartAngle->ValueChanged += gcnew EventHandler(this, &MyForm::OnStartAngleChanged); this->Controls->Add(nudStartAngle);
			UpdateBackendUiState();
			UpdateTrajectoryUiState();
		}

		void UpdateBackendUiState()
		{
			bool tracIkSelected = (cbBackend->SelectedIndex == 1);
			bool obstaclesEnabled = !tracIkSelected;
			btnGenerateObstacles->Enabled = obstaclesEnabled;
			btnClearObstacles->Enabled = obstaclesEnabled;

			if (tracIkSelected)
			{
				btnGenerateObstacles->Text = L"Препятствия отключены";
				btnGenerateObstacles->ForeColor = Color::Gold;
				btnGenerateObstacles->BackColor = SystemColors::Control;
				btnClearObstacles->BackColor = SystemColors::Control;

				if (obstacleX->Count > 0)
				{
					obstacleX->Clear();
					obstacleY->Clear();
					obstacleHalf->Clear();
					ClearTrajectoryCache();
					this->Invalidate();
				}
			}
			else
			{
				btnGenerateObstacles->Text = L"Сгенерировать препятствия";
				btnGenerateObstacles->ForeColor = SystemColors::ControlDarkDark;
				btnGenerateObstacles->BackColor = SystemColors::Info;
				btnClearObstacles->BackColor = SystemColors::Info;
			}
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

			if (trajectoryMode)
			{
				lblStartAngle->Location = Point(245, 194);
				lblStartAngle->Width = 200;
				nudStartAngle->Location = Point(245, 226);
				nudStartAngle->Width = 200;
				lblStartAngle->Visible = true;
				nudStartAngle->Visible = true;
				lblStartAngle->Enabled = true;
				nudStartAngle->Enabled = true;
			}
			else
			{
				lblStartAngle->Location = Point(20, 340);
				lblStartAngle->Width = 200;
				nudStartAngle->Location = Point(20, 372);
				nudStartAngle->Width = 200;
				lblStartAngle->Visible = false;
				nudStartAngle->Visible = false;
				lblStartAngle->Enabled = false;
				nudStartAngle->Enabled = false;
			}

			if (!trajectoryMode)
			{
				ClearTrajectoryCache();
				nudStartAngle->Value = Decimal(0);
			}

			this->Invalidate();
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
			lengths->Add((float)nudBaseLength->Value);
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
			if (animationTimer)
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
			return (float)(rngState) * 2.3283064e-10f;
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

		void SetDefaultStartPoint(bool forceResetCustomization)
		{
			if (forceResetCustomization)
				startPointCustomized = false;

			if (startPointCustomized)
				return;

			syncingDefaultStartPoint = true;
			nudStartX->Value = Decimal((double)((float)nudTargetX->Value * 0.5f));
			nudStartY->Value = Decimal((double)((float)nudTargetY->Value * 0.5f));
			syncingDefaultStartPoint = false;
		}

		array<float>^ BuildObstacleBuffer()
		{
			int count = obstacleX->Count;
			array<float>^ data = gcnew array<float>(count * 3);

			for (int i = 0; i < count; ++i)
			{
				data[3 * i] = obstacleX[i];
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
			float baseLength = (float)nudBaseLength->Value;
			float stretchFactor = (float)nudStretchFactor->Value;
			float tx = (float)nudTargetX->Value;
			float ty = (float)nudTargetY->Value;

			float dist = (float)Math::Sqrt(tx * tx + ty * ty);

			if (dist <= 1e-6f)
			{
				ClearTrajectoryCache();
				this->Invalidate();
				return;
			}

			float maxReach = nSegments * (varLen ? baseLength * stretchFactor : baseLength);
			float slack = maxReach - dist;
			float halfMin = Math::Max(0.14f * baseLength, 0.12f);
			float halfMax = Math::Min(0.26f * baseLength + 0.05f * (slack / (baseLength + 1e-6f)), 0.32f);

			if (halfMax < halfMin + 0.03f)
				halfMax = halfMin + 0.03f;

			float alongMargin = Math::Max(0.55f * baseLength, 1.5f * (halfMax + ObstacleClearance));
			alongMargin = Math::Max(alongMargin, 0.35f);

			float usableLen = dist - 2.0f * alongMargin;
			int obstacleCountToCreate = 2 + (int)(Rand01() * 3.0f);

			if (nSegments == 2 && obstacleCountToCreate > 3)
				obstacleCountToCreate = 3;

			if (slack < 0.35f * baseLength && obstacleCountToCreate > 2)
				obstacleCountToCreate = 2;

			while (obstacleCountToCreate > 2)
			{
				float gapTest = usableLen / (obstacleCountToCreate + 1);

				if (gapTest >= 2.2f * halfMax)
					break;

				--obstacleCountToCreate;
			}

			float ux = tx / dist;
			float uy = ty / dist;
			float nx = -uy;
			float ny = ux;
			float gap = usableLen / (obstacleCountToCreate + 1);
			float firstSide = (Rand01() < 0.5f) ? -1.0f : 1.0f;

			for (int i = 0; i < obstacleCountToCreate; ++i)
			{
				float nominalS = alongMargin + gap * (i + 1);
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
			String^ value = currentEps.ToString("E3", System::Globalization::CultureInfo::InvariantCulture);

			if (txtEps->Text != value)
				txtEps->Text = value;
		}

		void OnEpsTextChanged(Object^ sender, EventArgs^ e)
		{
			float val;

			if (float::TryParse(txtEps->Text, System::Globalization::NumberStyles::Float, System::Globalization::CultureInfo::InvariantCulture, val))
			{
				if (val < 1e-9f)
					val = 1e-9f;

				if (val > 1e-1f)
					val = 1e-1f;

				currentEps = val;
				UpdateEpsDisplay();
				OnAnyChanged(nullptr, nullptr);
			}
			else
			{
				UpdateEpsDisplay();
			}
		}

		void OnEpsOrderUp(Object^ sender, EventArgs^ e)
		{
			float newVal = currentEps * 10.0f;

			if (newVal > 1e-1f)
				newVal = 1e-1f;

			currentEps = newVal;
			UpdateEpsDisplay();
			OnAnyChanged(nullptr, nullptr);
		}

		void OnEpsOrderDown(Object^ sender, EventArgs^ e)
		{
			float newVal = currentEps / 10.0f;

			if (newVal < 1e-9f)
				newVal = 1e-9f;

			currentEps = newVal;
			UpdateEpsDisplay();
			OnAnyChanged(nullptr, nullptr);
		}

		void OnAngleChanged(Object^ sender, EventArgs^ e)
		{
			if (updatingFromMouse)
				return;

			float val = WrapPi((float)nudTargetAngle->Value);
			Decimal wrapped = Decimal((double)val);

			if (nudTargetAngle->Value != wrapped)
				nudTargetAngle->Value = wrapped;

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnStartAngleChanged(Object^ sender, EventArgs^ e)
		{
			if (updatingFromMouse)
				return;

			float val = WrapPi((float)nudStartAngle->Value);
			Decimal wrapped = Decimal((double)val);

			if (nudStartAngle->Value != wrapped)
				nudStartAngle->Value = wrapped;

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnClearObstaclesClick(Object^ sender, EventArgs^ e)
		{
			ClearObstacles();
		}

		void OnGenerateObstaclesClick(Object^ sender, EventArgs^ e)
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

		void ComputeEndEffector(array<float>^ poseAngles, array<float>^ poseLengths, float% outX, float% outY, float% outA)
		{
			float x = 0;
			float y = 0;
			float phi = 0;
			int count = Math::Min(poseAngles->Length, poseLengths->Length);

			for (int i = 0; i < count; ++i)
			{
				phi += poseAngles[i];
				x += poseLengths[i] * (float)Math::Cos(phi);
				y += poseLengths[i] * (float)Math::Sin(phi);
			}

			outX = x;
			outY = y;
			outA = WrapPi(phi);
		}

		PoseSnapshot^ RunAgpCppAtPoint(float tx, float ty, float ta)
		{
			variableLengths = cbVarLen->Checked;

			float maxTheta = (float)nudMaxTheta->Value;
			int maxIter = (int)nudMaxIter->Value;
			bool adaptive = cbAdaptive->Checked;
			float r_param = (float)nudR->Value;
			float eps = currentEps;
			unsigned int seed = (unsigned int)GetTickCount();
			float baseLength = (float)nudBaseLength->Value;
			float stretchFactor = (float)nudStretchFactor->Value;

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
				nSegments,
				variableLengths,
				maxTheta,
				tx,
				ty,
				ta,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObstacleData,
				obstacleCount,
				0,
				nullptr,
				nullptr
			);

			float* bestQ;
			float bestX;
			float bestY;
			float bestA;
			float bestF;
			size_t actualIterations;
			float achievedEps;

			LARGE_INTEGER t0;
			LARGE_INTEGER t1;
			LARGE_INTEGER fq;

			QueryPerformanceFrequency(&fq);
			QueryPerformanceCounter(&t0);

			fManip(
				nSegments,
				variableLengths,
				maxTheta,
				tx,
				ty,
				ta,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObstacleData,
				obstacleCount,
				&bestQ,
				&bestX,
				&bestY,
				&bestA,
				&bestF,
				&actualIterations,
				&achievedEps,
				0,
				nullptr
			);

			QueryPerformanceCounter(&t1);

			PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);

			for (int i = 0; i < nSegments; ++i)
				pose->Angles[i] = bestQ[i];

			if (variableLengths)
			{
				for (int i = 0; i < nSegments; ++i)
					pose->Lengths[i] = bestQ[nSegments + i];
			}
			else
			{
				for (int i = 0; i < nSegments; ++i)
					pose->Lengths[i] = baseLength;
			}

			Marshal::FreeCoTaskMem(IntPtr(bestQ));

			pose->EndX = bestX;
			pose->EndY = bestY;
			pose->EndA = WrapPi(bestA);
			pose->BestF = bestF;
			pose->Iterations = (int)actualIterations;
			pose->AchievedEps = achievedEps;
			pose->Millis =
				(float)(
					1.0e3 *
					(double)(t1.QuadPart - t0.QuadPart) /
					(double)fq.QuadPart
					);

			return pose;
		}

		PoseSnapshot^ RunTracIkPositioning(float tx, float ty, float ta)
		{
			float baseLength = (float)nudBaseLength->Value;
			TracIkRunner* runner = GetTracIkRunner();

			double bestF = 0.0;
			double bestX = 0.0;
			double bestY = 0.0;
			double bestA = 0.0;
			double millis = 0.0;

			runner->Solve(
				tx,
				ty,
				ta,
				bestF,
				bestX,
				bestY,
				bestA,
				millis
			);

			const KDL::JntArray& qOut = runner->GetSolution();

			PoseSnapshot^ pose = gcnew PoseSnapshot(nSegments);

			for (int i = 0; i < nSegments; ++i)
			{
				pose->Angles[i] = (float)qOut(i);
				pose->Lengths[i] = baseLength;
			}

			pose->EndX = (float)bestX;
			pose->EndY = (float)bestY;
			pose->EndA = (float)bestA;
			pose->BestF = (float)bestF;
			pose->Iterations = -1;
			pose->AchievedEps = -1.0f;
			pose->Millis = (float)millis;

			return pose;
		}

		void UpdatePositioningStats(PoseSnapshot^ pose, float tx, float ty, float ta, bool isTracIk)
		{
			float dx = pose->EndX - tx;
			float dy = pose->EndY - ty;
			float distance = sqrtf(dx * dx + dy * dy);

			String^ iterationsStr = isTracIk ? "—" : pose->Iterations.ToString();
			String^ epsStr = isTracIk ? "—" : pose->AchievedEps.ToString("E3");

			lblInfo->Text = String::Format(
				L"Функционал: {0:F6}\nБлизость захвата: {1:F5}\nДостигнутая точка: ({2:F3}, {3:F3})\nДостигнутый угол: {4:F3} рад.\nВремя: {5:F3} мс\nЧисло шагов: {6}\nДостигнутая точность: {7}",
				pose->BestF,
				distance,
				pose->EndX,
				pose->EndY,
				pose->EndA,
				pose->Millis,
				iterationsStr,
				epsStr
			);
		}

		void UpdateTrajectoryStats(List<PoseSnapshot^>^ poses, float totalMillis, size_t totalIterations, float finalAchievedEps, bool isTracIk)
		{
			if (poses->Count == 0)
			{
				lblInfo->Text = L"Траектория не построена";
				return;
			}

			float totalEnergy = 0;

			for (int i = 1; i < poses->Count; ++i)
				totalEnergy += ComputeTransitionEnergy(poses[i - 1], poses[i]);

			PoseSnapshot^ lastPose = poses[poses->Count - 1];

			float rawTargetX = (float)nudTargetX->Value;
			float rawTargetY = (float)nudTargetY->Value;
			float finalDx = lastPose->EndX - rawTargetX;
			float finalDy = lastPose->EndY - rawTargetY;
			float finalDistance = sqrtf(finalDx * finalDx + finalDy * finalDy);

			int actualIntermediateCount = Math::Max(0, poses->Count - 2);

			String^ iterationsStr = isTracIk ? "—" : totalIterations.ToString();
			String^ epsStr = isTracIk ? "—" : finalAchievedEps.ToString("E3");

			lblInfo->Text = String::Format(
				L"Промежуточных точек: {0}\nФункционал траектории: {1:F6}\nБлизость финиша: {2:F5}\nДостигнутый угол: {3:F3} рад.\nВремя: {4:F3} мс\nЧисло шагов: {5}\nДостигнутая точность: {6}",
				actualIntermediateCount,
				totalEnergy,
				finalDistance,
				lastPose->EndA,
				totalMillis,
				iterationsStr,
				epsStr
			);
		}

		float ComputeTransitionEnergy(PoseSnapshot^ prevPose, PoseSnapshot^ nextPose)
		{
			float total = 0;
			float prevPrefix = 0;
			float nextPrefix = 0;

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
				RunTrajectoryWithTracIk();
			else
				RunTrajectoryWithAGP();
		}

		bool BuildPoseFromState(const float* state, bool varLen, float baseLength, PoseSnapshot^% pose)
		{
			pose = gcnew PoseSnapshot(nSegments);

			for (int i = 0; i < nSegments; ++i)
			{
				pose->Angles[i] = state[i];
				pose->Lengths[i] = varLen ? state[nSegments + i] : baseLength;
			}

			float x;
			float y;
			float a;

			ComputeEndEffector(pose->Angles, pose->Lengths, x, y, a);

			pose->EndX = x;
			pose->EndY = y;
			pose->EndA = a;

			return true;
		}

		void RunTrajectoryWithAGP()
		{
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

			float startX = (float)nudStartX->Value;
			float startY = (float)nudStartY->Value;
			float startA = (float)nudStartAngle->Value;
			float targetX = (float)nudTargetX->Value;
			float targetY = (float)nudTargetY->Value;
			float targetA = (float)nudTargetAngle->Value;
			float maxTheta = (float)nudMaxTheta->Value;
			float baseLength = (float)nudBaseLength->Value;
			float stretchFactor = (float)nudStretchFactor->Value;
			float r_param = (float)nudR->Value;
			float eps = currentEps;

			int maxIter = (int)nudMaxIter->Value;
			bool varLen = cbVarLen->Checked;
			bool adaptive = cbAdaptive->Checked;
			unsigned int seed = (unsigned int)GetTickCount();
			int stateDim = nSegments << 1;

			pStart(
				nSegments,
				varLen,
				maxTheta,
				startX,
				startY,
				startA,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				0,
				nullptr,
				nullptr
			);

			LARGE_INTEGER totalT0;
			LARGE_INTEGER totalT1;
			LARGE_INTEGER fq;

			QueryPerformanceFrequency(&fq);
			QueryPerformanceCounter(&totalT0);

			float* startQ;
			float startXOut;
			float startYOut;
			float startAOut;
			float startF;
			size_t startIterations;
			float startEps;

			fManip(
				nSegments,
				varLen,
				maxTheta,
				startX,
				startY,
				startA,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				&startQ,
				&startXOut,
				&startYOut,
				&startAOut,
				&startF,
				&startIterations,
				&startEps,
				0,
				nullptr
			);

			PoseSnapshot^ startPose;
			BuildPoseFromState(startQ, varLen, baseLength, startPose);

			startPose->EndX = startXOut;
			startPose->EndY = startYOut;
			startPose->EndA = WrapPi(startAOut);
			startPose->BestF = startF;
			startPose->Iterations = (int)startIterations;
			startPose->AchievedEps = startEps;

			std::vector<float> startState(stateDim);

			for (int i = 0; i < nSegments; ++i)
			{
				startState[i] = startPose->Angles[i];
				startState[nSegments + i] = startPose->Lengths[i];
			}

			Marshal::FreeCoTaskMem(IntPtr(startQ));

			pStart(
				nSegments,
				varLen,
				maxTheta,
				targetX,
				targetY,
				targetA,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				1,
				startState.data(),
				nullptr
			);

			float* finalQ;
			float finalXOut;
			float finalYOut;
			float finalAOut;
			float finalF;
			size_t finalIterations;
			float finalEps;

			fManip(
				nSegments,
				varLen,
				maxTheta,
				targetX,
				targetY,
				targetA,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				&finalQ,
				&finalXOut,
				&finalYOut,
				&finalAOut,
				&finalF,
				&finalIterations,
				&finalEps,
				1,
				startState.data()
			);

			PoseSnapshot^ finalPose;
			BuildPoseFromState(finalQ, varLen, baseLength, finalPose);

			finalPose->EndX = finalXOut;
			finalPose->EndY = finalYOut;
			finalPose->EndA = WrapPi(finalAOut);
			finalPose->BestF = finalF;
			finalPose->Iterations = (int)finalIterations;
			finalPose->AchievedEps = finalEps;

			std::vector<float> finalState(stateDim);

			for (int i = 0; i < nSegments; ++i)
			{
				finalState[i] = finalPose->Angles[i];
				finalState[nSegments + i] = finalPose->Lengths[i];
			}

			Marshal::FreeCoTaskMem(IntPtr(finalQ));

			pStart(
				nSegments,
				varLen,
				maxTheta,
				0,
				0,
				0,
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				2,
				startState.data(),
				finalState.data()
			);

			float* trajPoints;
			int pointCount;
			size_t trajectoryIterations;

			pBuildTrajectory(
				nSegments,
				varLen,
				maxTheta,
				startState.data(),
				finalState.data(),
				maxIter,
				r_param,
				adaptive,
				eps,
				seed,
				baseLength,
				stretchFactor,
				pObs,
				obstacleCount,
				&trajPoints,
				&pointCount,
				&trajectoryIterations
			);

			QueryPerformanceCounter(&totalT1);

			float millis =
				(float)(
					1.0e3 *
					(double)(totalT1.QuadPart - totalT0.QuadPart) /
					(double)fq.QuadPart
					);

			size_t totalIterations =
				trajectoryIterations +
				startIterations +
				finalIterations;

			plannedPoses->Clear();

			for (int i = 0; i < pointCount; ++i)
			{
				const float* ptr = trajPoints + i * stateDim;
				PoseSnapshot^ pose;
				BuildPoseFromState(ptr, varLen, baseLength, pose);
				plannedPoses->Add(pose);
			}

			Marshal::FreeCoTaskMem(IntPtr(trajPoints));

			plannedPoses[0]->EndX = startPose->EndX;
			plannedPoses[0]->EndY = startPose->EndY;
			plannedPoses[0]->EndA = startPose->EndA;

			plannedPoses[plannedPoses->Count - 1]->EndX = finalPose->EndX;
			plannedPoses[plannedPoses->Count - 1]->EndY = finalPose->EndY;
			plannedPoses[plannedPoses->Count - 1]->EndA = finalPose->EndA;

			UpdateTrajectoryStats(
				plannedPoses,
				millis,
				totalIterations,
				finalEps,
				false
			);

			BuildAnimationFramesFromPlan();
			ApplyPoseToManipulator(plannedPoses[0]);
			StartAnimationIfNeeded();
			this->Invalidate();
		}

		void RunTrajectoryWithTracIk()
		{
			ClearTrajectoryCache();

			float startX = (float)nudStartX->Value;
			float startY = (float)nudStartY->Value;
			float startA = (float)nudStartAngle->Value;
			float targetX = (float)nudTargetX->Value;
			float targetY = (float)nudTargetY->Value;
			float targetA = (float)nudTargetAngle->Value;

			PoseSnapshot^ startPose =
				RunTracIkPositioning(
					startX,
					startY,
					startA
				);

			PoseSnapshot^ endPose =
				RunTracIkPositioning(
					targetX,
					targetY,
					targetA
				);

			float totalMillis =
				startPose->Millis +
				endPose->Millis;

			plannedPoses->Clear();
			plannedPoses->Add(startPose);
			plannedPoses->Add(endPose);

			UpdateTrajectoryStats(
				plannedPoses,
				totalMillis,
				0,
				-1.0f,
				true
			);

			BuildAnimationFramesFromPlan();
			ApplyPoseToManipulator(plannedPoses[0]);
			StartAnimationIfNeeded();
			this->Invalidate();
		}

		void RunPositioningMode()
		{
			StopAnimation();

			float tx = (float)nudTargetX->Value;
			float ty = (float)nudTargetY->Value;
			float ta = (float)nudTargetAngle->Value;

			bool isTracIk = (cbBackend->SelectedIndex == 1);

			PoseSnapshot^ pose =
				isTracIk
				? RunTracIkPositioning(tx, ty, ta)
				: RunAgpCppAtPoint(tx, ty, ta);

			ClearTrajectoryCache();
			ApplyPoseToManipulator(pose);
			UpdatePositioningStats(pose, tx, ty, ta, isTracIk);
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
				float t = s / 20.0f;
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
			if (trajectoryPathWorld->Count < 2)
				return;

			for (int i = 1; i < trajectoryPathWorld->Count; ++i)
				DrawTrajectorySegmentClipped(g, trajectoryPathWorld[i - 1], trajectoryPathWorld[i]);

			if (plannedPoses->Count > 2)
			{
				for (int i = 1; i < plannedPoses->Count - 1; ++i)
				{
					if (!IsDisplayPointAllowed(plannedPoses[i]->EndX, plannedPoses[i]->EndY))
						continue;

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

					if (trajectoryPathWorld->Count == 0 ||
						DistanceSquared(trajectoryPathWorld[trajectoryPathWorld->Count - 1], p) > 1e-10f)
					{
						trajectoryPathWorld->Add(p);
					}
				}

				return;
			}

			for (int i = 0; i < plannedPoses->Count; ++i)
			{
				PointF p(plannedPoses[i]->EndX, plannedPoses[i]->EndY);

				if (trajectoryPathWorld->Count == 0 ||
					DistanceSquared(trajectoryPathWorld[trajectoryPathWorld->Count - 1], p) > 1e-10f)
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
				PoseSnapshot^ firstPose = plannedPoses[i];
				PoseSnapshot^ secondPose = plannedPoses[i + 1];

				for (int frame = 1; frame <= AnimationFramesPerSegment; ++frame)
				{
					float t = frame / (float)AnimationFramesPerSegment;
					PoseSnapshot^ interp = gcnew PoseSnapshot(nSegments);

					for (int j = 0; j < nSegments; ++j)
					{
						interp->Angles[j] =
							LerpWrappedAngle(
								firstPose->Angles[j],
								secondPose->Angles[j],
								t
							);

						interp->Lengths[j] =
							variableLengths
							? firstPose->Lengths[j] +
							t * (secondPose->Lengths[j] - firstPose->Lengths[j])
							: (float)nudBaseLength->Value;
					}

					float x;
					float y;
					float angle;

					ComputeEndEffector(
						interp->Angles,
						interp->Lengths,
						x,
						y,
						angle
					);

					interp->EndX = x;
					interp->EndY = y;
					interp->EndA = angle;

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

		void OnAnimationTick(Object^ sender, EventArgs^ e)
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

			if (animationFrameIndex >= animationFrames->Count)
				StopAnimation();
		}

		PointF WorldToPixel(float wx, float wy)
		{
			PointF basePoint = GetBasePoint();
			return PointF(basePoint.X + wx * 160.0f, basePoint.Y - wy * 160.0f);
		}

		float GetCurrentMaxReach()
		{
			float baseLength = (float)nudBaseLength->Value;
			float stretchFactor = (float)nudStretchFactor->Value;

			return nSegments *
				(cbVarLen->Checked
					? baseLength * stretchFactor
					: baseLength);
		}

		bool IsDisplayPointAllowed(float x, float y)
		{
			float maxReach = GetCurrentMaxReach();

			if (x * x + y * y >
				(maxReach + 0.05f) *
				(maxReach + 0.05f))
			{
				return false;
			}

			for (int i = 0; i < obstacleX->Count; ++i)
			{
				float half = obstacleHalf[i] + ObstacleClearance;

				if (x >= obstacleX[i] - half &&
					x <= obstacleX[i] + half &&
					y >= obstacleY[i] - half &&
					y <= obstacleY[i] + half)
				{
					return false;
				}
			}

			return true;
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		void OnMouseDownPoint(Object^ sender, MouseEventArgs^ e)
		{
			PointF basePoint = GetBasePoint();
			float baseX = basePoint.X;
			float baseY = basePoint.Y;

			float targetX = (float)nudTargetX->Value;
			float targetY = (float)nudTargetY->Value;
			float targetA = (float)nudTargetAngle->Value;

			float pixelTargetX = baseX + targetX * 160.0f;
			float pixelTargetY = baseY - targetY * 160.0f;
			float angleLineLen = 25.0f;

			float endX = pixelTargetX + angleLineLen * (float)Math::Cos(targetA);
			float endY = pixelTargetY - angleLineLen * (float)Math::Sin(targetA);

			float dx = e->X - endX;
			float dy = e->Y - endY;

			if (dx * dx + dy * dy < 225.0f)
			{
				StopAnimation();
				activeDragHandle = DragHandle::AngleTarget;
				angleTargetDrag = true;
				updatingFromMouse = true;
				this->Capture = true;
				return;
			}

			float bestDistSquared = 100.0f;
			activeDragHandle = DragHandle::None;

			if (IsTrajectoryMode())
			{
				float startX = (float)nudStartX->Value;
				float startY = (float)nudStartY->Value;
				float startA = (float)nudStartAngle->Value;

				float pixelStartX = baseX + startX * 160.0f;
				float pixelStartY = baseY - startY * 160.0f;

				float sEndX = pixelStartX + angleLineLen * (float)Math::Cos(startA);
				float sEndY = pixelStartY - angleLineLen * (float)Math::Sin(startA);

				dx = e->X - sEndX;
				dy = e->Y - sEndY;

				if (dx * dx + dy * dy < 225.0f)
				{
					StopAnimation();
					activeDragHandle = DragHandle::AngleStart;
					angleStartDrag = true;
					updatingFromMouse = true;
					this->Capture = true;
					return;
				}

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

			if (targetDistSquared < bestDistSquared)
				activeDragHandle = DragHandle::Target;

			if (activeDragHandle != DragHandle::None)
			{
				StopAnimation();
				updatingFromMouse = true;
				this->Capture = true;
			}
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		void OnMouseMovePoint(Object^ sender, MouseEventArgs^ e)
		{
			if (activeDragHandle == DragHandle::None &&
				!angleTargetDrag &&
				!angleStartDrag)
			{
				return;
			}

			PointF basePoint = GetBasePoint();
			float baseX = basePoint.X;
			float baseY = basePoint.Y;

			if (angleTargetDrag)
			{
				float targetX = (float)nudTargetX->Value;
				float targetY = (float)nudTargetY->Value;

				float pixelTargetX = baseX + targetX * 160.0f;
				float pixelTargetY = baseY - targetY * 160.0f;

				float dx = e->X - pixelTargetX;
				float dy = e->Y - pixelTargetY;

				float angle = WrapPi(atan2f(-dy, dx));

				nudTargetAngle->Value = Decimal((double)angle);

				ClearTrajectoryCache();
				this->Invalidate();
				return;
			}

			if (angleStartDrag && IsTrajectoryMode())
			{
				float startX = (float)nudStartX->Value;
				float startY = (float)nudStartY->Value;

				float pixelStartX = baseX + startX * 160.0f;
				float pixelStartY = baseY - startY * 160.0f;

				float dx = e->X - pixelStartX;
				float dy = e->Y - pixelStartY;

				float angle = WrapPi(atan2f(-dy, dx));

				nudStartAngle->Value = Decimal((double)angle);

				ClearTrajectoryCache();
				this->Invalidate();
				return;
			}

			if (activeDragHandle == DragHandle::None)
				return;

			float minX = (float)nudTargetX->Minimum;
			float maxX = (float)nudTargetX->Maximum;
			float minY = (float)nudTargetY->Minimum;
			float maxY = (float)nudTargetY->Maximum;

			float newX = (e->X - baseX) / 160.0f;
			float newY = (baseY - e->Y) / 160.0f;

			newX = Math::Max(minX, Math::Min(maxX, newX));
			newY = Math::Max(minY, Math::Min(maxY, newY));

			if (activeDragHandle == DragHandle::Target)
			{
				nudTargetX->Value = Decimal((double)newX);
				nudTargetY->Value = Decimal((double)newY);
			}
			else if (activeDragHandle == DragHandle::Start)
			{
				startPointCustomized = true;
				nudStartX->Value = Decimal((double)newX);
				nudStartY->Value = Decimal((double)newY);
			}

			ClearTrajectoryCache();
			this->Invalidate();
		}

		[MethodImpl(MethodImplOptions::AggressiveInlining)]
		void OnMouseUpPoint(Object^ sender, MouseEventArgs^ e)
		{
			if (activeDragHandle != DragHandle::None ||
				angleTargetDrag ||
				angleStartDrag)
			{
				activeDragHandle = DragHandle::None;
				angleTargetDrag = false;
				angleStartDrag = false;
				updatingFromMouse = false;
				this->Capture = false;
			}
		}

		void OnBackendChanged(Object^ sender, EventArgs^ e)
		{
			UpdateBackendUiState();
			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnDemoModeChanged(Object^ sender, EventArgs^ e)
		{
			demoMode =
				(cbDemoMode->SelectedIndex == 0)
				? DemoMode::Positioning
				: DemoMode::TrajectoryPlanning;

			if (demoMode == DemoMode::TrajectoryPlanning &&
				!startPointCustomized)
			{
				SetDefaultStartPoint(false);
			}

			UpdateTrajectoryUiState();
		}

		void OnResize(Object^ sender, EventArgs^ e)
		{
			this->Invalidate();
		}

		void OnAnyChanged(Object^ sender, EventArgs^ e)
		{
			if (updatingFromMouse)
				return;

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnTargetChanged(Object^ sender, EventArgs^ e)
		{
			if (updatingFromMouse)
				return;

			if (!startPointCustomized)
				SetDefaultStartPoint(false);

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnStartPointChanged(Object^ sender, EventArgs^ e)
		{
			if (updatingFromMouse)
				return;

			if (!syncingDefaultStartPoint)
				startPointCustomized = true;

			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnAddClick(Object^ sender, EventArgs^ e)
		{
			++nSegments;
			angles->Add(0.0f);
			lengths->Add((float)nudBaseLength->Value);
			ClearTrajectoryCache();
			this->Invalidate();
		}

		void OnRemClick(Object^ sender, EventArgs^ e)
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

		void OnOptimizeClick(Object^ sender, EventArgs^ e)
		{
			if (IsTrajectoryMode())
				RunTrajectoryPlanningMode();
			else
				RunPositioningMode();
		}

	protected:
		[MethodImpl(MethodImplOptions::AggressiveInlining)]
			virtual void OnPaint(PaintEventArgs^ e) override
		{
			Form::OnPaint(e);

			Graphics^ g = e->Graphics;
			g->SmoothingMode = SmoothingMode::HighQuality;
			g->Clear(this->BackColor);

			int drawHeight = Math::Max(0, this->ClientSize.Height - 180);
			System::Drawing::Rectangle drawArea(0, 180, this->ClientSize.Width, drawHeight);
			g->FillRectangle(Brushes::White, drawArea);

			float targetX = (float)nudTargetX->Value;
			float targetY = (float)nudTargetY->Value;
			float targetA = (float)nudTargetAngle->Value;

			PointF basePoint = GetBasePoint();
			int baseX = (int)basePoint.X;
			int baseY = (int)basePoint.Y;

			float pixelTargetX = baseX + targetX * 160.0f;
			float pixelTargetY = baseY - targetY * 160.0f;

			g->DrawLine(wallPen, baseX - 25, baseY + 8, baseX + 25, baseY + 8);
			g->FillRectangle(wallHatchBrush, baseX - 25, baseY + 8, 50, 12);

			g->DrawLine(dashedPen, pixelTargetX - 25, pixelTargetY + 8, pixelTargetX + 25, pixelTargetY + 8);
			g->FillRectangle(wallHatchBrush, (int)pixelTargetX - 25, (int)pixelTargetY + 8, 50, 12);
			g->DrawEllipse(targetPen, pixelTargetX - 8.0f, pixelTargetY - 8.0f, 16.0f, 16.0f);

			float angleLen = 25.0f;
			float endX = pixelTargetX + angleLen * (float)Math::Cos(targetA);
			float endY = pixelTargetY - angleLen * (float)Math::Sin(targetA);

			g->DrawLine(angleTargetPen, pixelTargetX, pixelTargetY, endX, endY);

			if (IsTrajectoryMode())
			{
				float startX = (float)nudStartX->Value;
				float startY = (float)nudStartY->Value;
				float startA = (float)nudStartAngle->Value;

				float pixelStartX = baseX + startX * 160.0f;
				float pixelStartY = baseY - startY * 160.0f;

				g->DrawLine(dashedPen, pixelStartX - 25, pixelStartY + 8, pixelStartX + 25, pixelStartY + 8);
				g->FillRectangle(wallHatchBrush, (int)pixelStartX - 25, (int)pixelStartY + 8, 50, 12);
				g->DrawEllipse(startPen, pixelStartX - 8.0f, pixelStartY - 8.0f, 16.0f, 16.0f);

				float sEndX = pixelStartX + angleLen * (float)Math::Cos(startA);
				float sEndY = pixelStartY - angleLen * (float)Math::Sin(startA);

				g->DrawLine(angleStartPen, pixelStartX, pixelStartY, sEndX, sEndY);
			}

			if (trajectoryPathWorld->Count > 1 && IsTrajectoryMode())
				DrawTrajectoryPath(g);

			if (obstacleX->Count > 0)
			{
				array<float>^ xs = obstacleX->ToArray();
				array<float>^ ys = obstacleY->ToArray();
				array<float>^ hs = obstacleHalf->ToArray();

				int obstacleCount =
					Math::Min(
						xs->Length,
						Math::Min(
							ys->Length,
							hs->Length
						)
					);

				for (int obsIdx = 0; obsIdx < obstacleCount; ++obsIdx)
				{
					float cx = xs[obsIdx];
					float cy = ys[obsIdx];
					float half = hs[obsIdx];

					float left = (float)baseX + (cx - half) * 160.0f;
					float top = (float)baseY - (cy + half) * 160.0f;
					float size = 2.0f * half * 160.0f;

					float marginLeft = (float)baseX + (cx - half - ObstacleClearance) * 160.0f;
					float marginTop = (float)baseY - (cy + half + ObstacleClearance) * 160.0f;
					float marginSize = 2.0f * (half + ObstacleClearance) * 160.0f;

					g->DrawRectangle(obstacleMarginPen, marginLeft, marginTop, marginSize, marginSize);
					g->FillRectangle(obstacleBrush, left, top, size, size);
					g->DrawRectangle(obstaclePen, left, top, size, size);
				}
			}

			int drawableSegments =
				Math::Min(
					nSegments,
					Math::Min(
						angles->Count,
						lengths->Count
					)
				);

			array<PointF>^ pts =
				gcnew array<PointF>(
					drawableSegments + 1
				);

			pts[0] = PointF((float)baseX, (float)baseY);

			float x = 0.0f;
			float y = 0.0f;
			float phi = 0.0f;

			array<float>^ localAngles = angles->ToArray();
			array<float>^ localLengths = lengths->ToArray();

			for (int i = 0; i < drawableSegments; ++i)
			{
				phi += localAngles[i];
				x += localLengths[i] * (float)Math::Cos(phi);
				y += localLengths[i] * (float)Math::Sin(phi);

				pts[i + 1] =
					PointF(
						(float)baseX + x * 160.0f,
						(float)baseY - y * 160.0f
					);
			}

			for (int i = 0; i < drawableSegments; ++i)
				g->DrawLine(penRod, pts[i], pts[i + 1]);

			for (int i = 0; i <= drawableSegments; ++i)
				g->FillEllipse(jointBrush, pts[i].X - 8.0f, pts[i].Y - 8.0f, 16.0f, 16.0f);
		}
	};
}
