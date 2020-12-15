echo "==> Running all models on unmodified, original data."
echo "-> Warning: Don't assume cross validation recommendations are the best choice. Examine graphs to get optimal value."
echo ""
cd ../original_data/models/

echo ""
echo "========= KNN ========="
echo ""
python3 knn.py 

echo ""
echo "========= Linear Regression ========="
echo ""
python3 linear_regressor.py 

echo ""
echo "========= Ridge Regression ========="
echo ""
python3 ridge.py 

echo ""
echo "========= Lasso Regression ========="
echo ""
python3 lasso.py 

cd ../../
echo ""
echo "Finished."