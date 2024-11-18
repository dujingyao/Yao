#include<iostream>
#include<vector>
using namespace std;

// 插入排序函数
void insertionSort(vector<int>& arr) {
    int n=arr.size();
    for (int i=1;i<n;++i) {
        int key=arr[i]; // 要插入的元素
        int j=i-1;
        // 将比key大的元素向右移动
        while(j>=0&&arr[j]>key) {
            arr[j+1]=arr[j];
            j--;
        }
        arr[j+1]=key; // 插入正确的位置
    }
}
int main() {
    vector<int> array;
    int n;
    cout<<"请输入要排序的个数：";
    cin>>n;
    cout<<"请输入排序前的数字：";
    while(n--){
        int x;
        cin>>x;
        array.push_back(x);
    }

    // 执行插入排序
    insertionSort(array);

    // 排序后
    cout<<"排序后: ";
    for (int num : array) {
        cout<<num<<" ";
    }
    cout<<endl;

    return 0;
}