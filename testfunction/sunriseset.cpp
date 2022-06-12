#include <iostream>
#include <cmath>
using namespace std;

#define PI 3.1415926

//定义全局变量

int doy = 318;
//判断是否为闰年:若为闰年，返回1；若非闰年，返回0
int leap_year(int year)
{
	if ((year % 400 == 0) || ((year % 100 != 0) && (year % 4 == 0)))
		return 1;
	else
		return 0;
}

//求从格林威治时间公元2000年1月1日到计算日天数days
int GetDoy(int year, int doy)
{
	int i,a = 0;
	for (i = 2000; i < year; i++)
	{
		if (leap_year(i))
			a = a + 366;
		else
			a = a + 365;
	}
	a += doy;
	return a;
}

//判断并返回结果(日出)
double sunrise(double UT,double UTo, double glong, double glat,int yeardoy)
{
    //求格林威治时间公元2000年1月1日到计算日的世纪数t
    double t_century = (( double) yeardoy + UTo / 360) / 36525;
    //求太阳的平黄径
    double L_sun = (280.460 + 36000.770 * t_century);
    //求太阳的平近点角
    double G_sun = (357.528 + 35999.050 * t_century);
    //求黄道经度
    double ecliptic_longitude = (L_sun + 1.915 * sin(G_sun * PI / 180) 
                                        + 0.02 * sin(2 * G_sun * PI / 180));
    //求地球倾角
    double earth_tilt = (23.4393 - 0.0130 * t_century);
    //求太阳偏差
    double sun_deviation = (180 / PI* asin(sin(PI / 180 * earth_tilt)* sin(PI / 180 * ecliptic_longitude)));
    //求格林威治时间的太阳时间角GHA
    double GHA = (UTo - 180 - 1.915 * sin(G_sun * PI / 180)
                - 0.02 * sin(2 * G_sun * PI / 180)
                + 2.466 * sin(2 * ecliptic_longitude * PI / 180)
                - 0.053 * sin(4 * ecliptic_longitude * PI / 180));
    //求修正值e
    double h = -0.833;
    double e = 180 / PI* acos((sin(h * PI / 180)- sin(glat * PI / 180)* sin(sun_deviation * PI / 180))/ (cos(glat * PI / 180)* cos(sun_deviation * PI / 180)));
    //求日出时间
    double UT_rise = (UTo - (GHA + glong + e));

    double d;
	if (UT >= UTo)
	{d = UT - UTo;}
	else
	{d = UTo - UT;}

    if(d >= 0.1)
    {
        UTo = UT;
        UT = UT_rise;
    }

    return UT;
}

//判断并返回结果(日落)
double sunset(double UT,double UTo, double glong, double glat,int yeardoy)
{
    //求格林威治时间公元2000年1月1日到计算日的世纪数t
    double t_century = (( double) yeardoy + UTo / 360) / 36525;
    //求太阳的平黄径
    double L_sun = (280.460 + 36000.770 * t_century);
    //求太阳的平近点角
    double G_sun = (357.528 + 35999.050 * t_century);
    //求黄道经度
    double ecliptic_longitude = (L_sun + 1.915 * sin(G_sun * PI / 180) 
                                        + 0.02 * sin(2 * G_sun * PI / 180));
    //求地球倾角
    double earth_tilt = (23.4393 - 0.0130 * t_century);
    //求太阳偏差
    double sun_deviation = (180 / PI* asin(sin(PI / 180 * earth_tilt)* sin(PI / 180 * ecliptic_longitude)));
    //求格林威治时间的太阳时间角GHA
    double GHA = (UTo - 180 - 1.915 * sin(G_sun * PI / 180)
                - 0.02 * sin(2 * G_sun * PI / 180)
                + 2.466 * sin(2 * ecliptic_longitude * PI / 180)
                - 0.053 * sin(4 * ecliptic_longitude * PI / 180));
    //求修正值e
    double h = -0.833;
    double e = 180 / PI* acos((sin(h * PI / 180)- sin(glat * PI / 180)* sin(sun_deviation * PI / 180))/ (cos(glat * PI / 180)* cos(sun_deviation * PI / 180)));
    //求日出时间
    double UT_set = (UTo - (GHA + glong - e));

    // double d;
	// if (UT >= UTo)
	// {d = UT - UTo;}
	// else
	// {d = UTo - UT;}

    if(abs(UT - UTo) >= 0.1)
    {
        UTo = UT;
        UT = UT_set;
    }

    return UT;
}

//求时区
int Zone( double glong)
{
    int timeZone ;
    int shangValue = (int)(glong / 15);
    double yushuValue =  glong - shangValue*15;
//    printf("%lf\n",yushuValue);
    if (yushuValue <= 7.5) {
        timeZone = shangValue;
    } else {
        timeZone = shangValue +(glong > 0 ?  1 :-1);
    }
    return timeZone;
}

void output( double rise,  double set,  double glong)
{
	if ((int) (60 * (rise / 15 + Zone(glong) - (int) (rise / 15 + Zone(glong))))< 10)
	{
		printf("The time at which the sun rises is %d:0%d \n",(int) (rise / 15 + Zone(glong)),
				(int) (60* (rise / 15 + Zone(glong)- (int) (rise / 15 + Zone(glong)))));
	}
	else
	{
		printf("The time at which the sun rises is %d:%d \n",(int) (rise / 15 + Zone(glong)),
				(int) (60* (rise / 15 + Zone(glong)- (int) (rise / 15 + Zone(glong)))));
	}
 
	if ((int) (60 * (set / 15 + Zone(glong) - (int) (set / 15 + Zone(glong))))< 10)
	{
		printf("The time at which the sun sets is %d:%d \n",(int) (set / 15 + Zone(glong)),
				(int) (60* (set / 15 + Zone(glong)- (int) (set / 15 + Zone(glong)))));
	}
	else
	{
		printf("The time at which the sun sets is %d:%d \n",(int) (set / 15 + Zone(glong)),
				(int) (60* (set / 15 + Zone(glong)- (int) (set / 15 + Zone(glong)))));
	}
}



int main(void)
{
    int year=2000;
    int month=11;
    int date=13;
    double UTo = 180.0;

    // int result = days(year,month,date);
    int result = GetDoy(year, doy);
    std::cout<<result<<std::endl;

    double rise, set;
}

