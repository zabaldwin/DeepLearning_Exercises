#include <iostream>                                                                                                                                                                                   
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TRandom3.h>

using namespace std;

vector<double> generateNormalData(int numEvents) {
    TRandom3 randGen;
    vector<double> data;
    for (int i = 0; i < numEvents; ++i) {
        // Simulate normal response (i.e. Gaussian distribution)
        double response = randGen.Gaus(0, 1);
        data.push_back(response);
    }
    return data;
}

int main() {
    int numNormalEvents = 100000;

    vector<double> normalData = generateNormalData(numNormalEvents);

    TFile outputFile("normalData.root", "RECREATE");
    TTree tree("tree", "Particle Detector Responses");

    double response;
    tree.Branch("response", &response);

    for (const auto& dataPoint : normalData) {
        response = dataPoint;
        tree.Fill();
    }

    tree.Write();
    outputFile.Close();

    cout << "Normal data generation completed. Saved to normalData.root." << endl;

    TH1D histNormal("response_histogram_", "Response Distribution", 100, -10, 10);
    for (const auto& dataPoint : normalData) {
        histNormal.Fill(dataPoint);
    }

    TCanvas canvas("response_canvas_normal", "Response Histogram (Normal)", 800, 600);
    histNormal.Draw();
    canvas.SaveAs("response_histogram_normal.png");

    return 0;
}
