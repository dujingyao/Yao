#include<iostream>
#include<vector>
using namespace std;
vector<vector<int>> func(vector<vector<int>> a,vector<vector<int>> b,int n){
    vector<vector<int>> d(n,vector<int>(n));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            for(int k=0;k<n;k++){
                d[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    return d;
}
vector<vector<int>> matrixPower(vector<vector<int>> A, int m, int n) {
    vector<vector<int>> result(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        result[i][i] = 1; // 单位矩阵
    }

    while (m > 0) {
        if (m % 2 == 1) { // 如果m是奇数
            result = multiply(result, A, n);
        }
        A = multiply(A, A, n); // 平方A
        m /= 2; // m右移一位
    }

    return result;
}


int main(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> a(n,vector<int>(n));
    vector<vector<int>> b(n,vector<int>(n,0));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            scanf("%d",&a[i][j]);
        }
    }
    vector<vector<int>> result = matrixPower(a, m, n);

    for (const auto& row : result) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}