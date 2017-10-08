function [X] = mergeSolutions(rows, folder)

    files = dir([folder '*.dat']);
    l = length(files);

    solution = [];
    for file = files'
        name = load(file.name);
        m = length(name)/rows;
        for i = 1:m
            solution = [solution name(1:rows)];
            name(1:rows)=[];
        end
    end

    X = solution';

end