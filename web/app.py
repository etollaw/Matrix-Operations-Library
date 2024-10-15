from flask import Flask, render_template, request
import numpy as np
from matrix_operations.operations import (
    matrix_multiplication,
    determinant,
    eigenvalues_and_vectors,
    inverse_matrix,
    transpose_matrix
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        try:
            # Collecting data for Matrix 1
            rows1 = int(request.form['rows1'])
            cols1 = int(request.form['cols1'])

            matrix1 = []
            for i in range(rows1):
                row = [float(request.form[f'matrix1_row{i}_col{j}']) for j in range(cols1)]
                matrix1.append(row)

            # Collecting data for Matrix 2
            rows2 = int(request.form['rows2'])
            cols2 = int(request.form['cols2'])

            matrix2 = []
            for i in range(rows2):
                row = [float(request.form[f'matrix2_row{i}_col{j}']) for j in range(cols2)]
                matrix2.append(row)

            A = np.array(matrix1)
            B = np.array(matrix2)

            # Print matrices for debugging
            print("Matrix A:", A)
            print("Matrix B:", B)

            # Perform operation based on selected button
            operation = request.form.get('operation')
            if operation == 'multiply':
                if cols1 == rows2:
                    result = matrix_multiplication(A, B).tolist()
                    results['multiplication'] = {'result': result}
                else:
                    results['multiplication'] = "Invalid dimensions for multiplication"
            elif operation == 'determinant':
                if A.shape[0] == A.shape[1]:
                    results['determinant'] = determinant(A)
                else:
                    results['determinant'] = "Not applicable"
            elif operation == 'inverse':
                if A.shape[0] == A.shape[1] and determinant(A) != 0:
                    results['inverse'] = inverse_matrix(A).tolist()
                else:
                    results['inverse'] = "Not applicable"
            elif operation == 'transpose':
                results['transpose'] = transpose_matrix(A).tolist()
            elif operation == 'eigen':
                if A.shape[0] == A.shape[1]:
                    eigenvalues, eigenvectors = eigenvalues_and_vectors(A)
                    results['eigen'] = {'values': eigenvalues.tolist(), 'vectors': eigenvectors.tolist()}
                else:
                    results['eigen'] = "Not applicable"

        except Exception as e:
            results['error'] = f"Error: {str(e)}"

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
