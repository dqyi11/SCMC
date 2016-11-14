range = [[0.1 0.5]; [0.2, 0.6]];
X1 = unrestricted_sampling(1000, range);

figure;
scatter(X1(1,:),X1(2,:));

range = [[-0.5 0.5]; [-10., 10.]; [-1.0, 1.0]];
X2 = unrestricted_sampling(1000, range);

figure;
scatter3(X2(1,:),X2(2,:),X2(3,:));
