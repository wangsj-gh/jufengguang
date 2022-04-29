/* example3_2_1.c*/

// #include "stdafx.h"
// #include <math.h>
#include <cmath>
#include <iostream>

int main(void)
{
    int m, n, i, j, k;
    double a1, h, tau, r, *x, *t, **u, *a, *b, *c, *d, *ans;
    double phi(double x);
    double alpha(double t);
    double beta(double t);
    double f(double x, double t);
    double exact(double x, double t);
    double *chase_algorithm(double *a, double *b, double *c, int n, double *d);

    //  无条件稳定比较 (0.5,0.4) 处误差大约为4倍E(2h,4*tau)/E(h,tau)
    //	m=10;  //r=1
    //	n=100;
    m = 20; // r=1
    n = 400;
    a1 = 1.0;
    h = 1.0 / m;
    tau = 1.0 / n;
    r = a1 * tau / (h * h);
    std::cout << "r=" << r << std::endl;

    x = (double *)malloc(sizeof(double) * (m + 1));
    for (i = 0; i <= m; i++)
        x[i] = i * h;

    t = (double *)malloc(sizeof(double) * (n + 1));
    for (k = 0; k <= n; k++)
        t[k] = k * tau;

    u = (double **)malloc(sizeof(double *) * (m + 1));
    for (i = 0; i <= m; i++)
        u[i] = (double *)malloc(sizeof(double) * (n + 1));

    for (i = 0; i <= m; i++)
        u[i][0] = phi(x[i]); // initial condition
    for (k = 1; k <= n; k++)
    {
        u[0][k] = alpha(t[k]);
        u[m][k] = beta(t[k]);
    }

    a = (double *)malloc(sizeof(double) * (m - 1));
    b = (double *)malloc(sizeof(double) * (m - 1));
    c = (double *)malloc(sizeof(double) * (m - 1));
    d = (double *)malloc(sizeof(double) * (m - 1));
    ans = (double *)malloc(sizeof(double) * (m - 1));
    for (k = 1; k <= n; k++)
    {
        for (i = 0; i < m - 1; i++)
        {
            d[i] = u[i + 1][k - 1] + tau * f(x[i + 1], t[k]);
            a[i] = -r;
            b[i] = 1.0 + 2 * r;
            c[i] = a[i];
        }
        d[0] = d[0] + r * u[0][k];
        d[m - 2] = d[m - 2] + r * u[m][k];
        ans = chase_algorithm(a, b, c, m - 1, d);
        for (i = 0; i < m - 1; i++)
            u[i + 1][k] = ans[i];
    }

    for (j = 1; j <= 10; j++)
    {
        k = j * (n / 10);
        std::cout << "x=" << x[m / 2] << "\t"
                  << "t=" << t[k] << "\t"
                  << "numerical=" << u[m / 2][k] << "\t"
                  << "exact=" << exact(x[m / 2], t[k]) << "\t"
                  << "error=" << fabs(u[m / 2][k] - exact(x[m / 2], t[k])) << "\t"
                  << std::endl;
    }

    free(a);
    free(b);
    free(c);
    free(d);
}

double phi(double x)
{
    return exp(x);
}

double alpha(double t)
{
    return exp(t);
}

double beta(double t)
{
    return exp(1.0 + t);
}

double f(double x, double t)
{
    return 0;
}

double exact(double x, double t)
{
    return exp(x + t);
}

double *chase_algorithm(double *a, double *b, double *c, int n, double *d)
{
    double *ans, *g, *w, p;
    int i;
    ans = (double *)malloc(sizeof(double) * n);
    g = (double *)malloc(sizeof(double) * n);
    w = (double *)malloc(sizeof(double) * n);
    g[0] = d[0] / b[0];
    w[0] = c[0] / b[0];

    for (i = 1; i < n; i++)
    {
        p = b[i] - a[i] * w[i - 1];
        g[i] = (d[i] - a[i] * g[i - 1]) / p;
        w[i] = c[i] / p;
    }
    ans[n - 1] = g[n - 1];
    i = n - 2;
    do
    {
        ans[i] = g[i] - w[i] * ans[i + 1];
        i = i - 1;

    } while (i >= 0);
    free(g);
    free(w);
    return ans;
}
