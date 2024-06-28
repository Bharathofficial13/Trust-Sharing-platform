<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
if(isset($_POST['btn']))
{
$imgpath=$_FILES['file']['name'];
	$errors= array();
    $fname = $_FILES['file']['name'];
    $file_tmp =$_FILES['file']['tmp_name'];
    move_uploaded_file($_FILES['file']['tmp_name'],"images/".$fname);
$qry=mysql_query("update register set dob='$dob',work='$work',address='$from',image='$fname' where id='$uid'");
$qry;
if($qry)
{
?>
<script language="javascript">
	alert("Success");
	window.location.href="home.php";
	</script>
	<?php

}
else
{
?>
<script language="javascript">
	alert("Unsuccess");
	window.location.href="profile.php";
	</script>
	<?php
}
}
?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Facebook Home</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/font-awesome.min.css" rel="stylesheet">
	<link href="css/datepicker3.css" rel="stylesheet">
	<link href="css/styles.css" rel="stylesheet">
	
	<!--Custom Font-->
	<link href="https://fonts.googleapis.com/css?family=Montserrat:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">
	<!--[if lt IE 9]>
	<script src="js/html5shiv.js"></script>
	<script src="js/respond.min.js"></script>
	<![endif]-->
    <style type="text/css">
<!--
.im {border-radius: 50%;
}
.im1 {  border-radius: 50%;
}
.im2 {  border-radius: 50%;
}
.style1 {color: #000000}
-->
    </style>
</head>
<body>
	<nav class="navbar navbar-custom navbar-fixed-top" role="navigation">
		<div class="container-fluid">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#sidebar-collapse"><span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span></button>
				<a class="navbar-brand" href="#"><span>Facebook</span>Home</a>
				<ul class="nav navbar-top-links navbar-right">
				</ul>
			</div>
		</div><!-- /.container-fluid -->
	</nav>
	<div id="sidebar-collapse" class="col-sm-3 col-lg-2 sidebar">
		<div class="profile-sidebar">
			<div class="profile-userpic">
			<?php
					$qry=mysql_query("select * from register where id='$uid'");
					$rw=mysql_fetch_array($qry); 
					?>
				
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>&nbsp;&nbsp;&nbsp;&nbsp;<?php echo $rw['name'];?></div>
			</div>
			<div class="clear"></div>
		</div>
		<div class="divider"></div>
		<form role="search">
			<div class="form-group">
				<input type="text" class="form-control" placeholder="Search">
			</div>
		</form>
		<ul class="nav menu">
			<li><a href="home.php"><i class="fa fa-home"></i> <span>Home</span></a></li>
										 <li><a href="frlist.php"><i class="fa fa-book"></i> <span>Friend List</span></a></li>
										<li><a href="list.php"><i class="fa fa-users"></i> <span>List</span></a></li>
									<li><a href="post.php"><i class=" fa fa-book"></i> <span>Post</span></a></li>
									<li><a href="accept.php"><i class="fa fa-users"></i> <span>Accept</span></a></li>
									<li><a href="group.php"><i class="fa fa-users"></i> <span>Create Group</span></a></li>
									<li><a href="grouppost.php"><i class="fa fa-users"></i> <span>Group Post</span></a></li>
									
									<li><a href="profile.php"><i class="fa fa-bell-o"></i> <span>Profile</span></a></li>
									<li><a href="index.php"><i class="fa fa-power-switch"></i> <span>Logout</span></a></li>
		</ul>
	</div><!--/.sidebar-->
		
	<div class="col-sm-9 col-sm-offset-3 col-lg-10 col-lg-offset-2 main">
		<div class="row">
			<ol class="breadcrumb">
				<li><a href="#">
					<em class="fa fa-home"></em>
				</a></li>
				<li class="active">Forms</li>
			</ol>
		</div><!--/.row-->
		
		<div class="row">
		  <div class="col-lg-12"></div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading"></div>
				  <div class="panel-body">
				      <form action="" method="post" enctype="multipart/form-data" name="form1">
                                  <div class="graph-visual tables-main">
                                    <h2 class="inner-tittle"><span class="sub-tittle">Profile Details </span></h2>
                                    <div class="graph">
                                      <div class="tables">
                                        <table width="80%" border="0">
                                          <tr>
                                            <td width="7%">&nbsp;</td>
                                            <td width="31%">&nbsp;</td>
                                            <td width="41%">&nbsp;</td>
                                            <td width="7%">&nbsp;</td>
                                            <td width="7%">&nbsp;</td>
                                            <td width="7%">&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td height="41">&nbsp;</td>
                                            <td><strong>Profile Photo </strong></td>
                                            <td><label>
                                            <input type="file" name="file">
                                            </label></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td height="36">&nbsp;</td>
                                            <td><strong>DOB</strong></td>
                                            <td><label>
                                              <input name="dob" type="date" id="dob">
                                            </label></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td height="36">&nbsp;</td>
                                            <td><strong>Name</strong></td>
                                            <td><label>
                                              <input name="name" type="text" id="name">
                                            </label></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td height="35">&nbsp;</td>
                                            <td><strong>Works</strong></td>
                                            <td><input name="work" type="text" id="work"></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td height="34">&nbsp;</td>
                                            <td><strong>From</strong></td>
                                            <td><input name="from" type="text" id="from"></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td><label>
                                              <input name="btn" type="submit" id="btn" value="Submit">
                                              <input type="submit" name="Submit2" value="Reset">
                                            </label></td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                          <tr>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                            <td>&nbsp;</td>
                                          </tr>
                                        </table>
                                      </div>
                                    </div>
                                    <h3 class="inner-tittle two">&nbsp;</h3>
                        </div>
                    </form>
                               
				  </div>
				</div><!-- /.panel-->
			</div>
			<!-- /.panel-->
	  </div><!-- /.col-->
			<div class="col-sm-12">
				<p class="back-link">Lumino Theme by <a href="https://www.medialoot.com">Medialoot</a></p>
			</div>
</div><!-- /.row -->
	</div><!--/.main-->
	
<script src="js/jquery-1.11.1.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/chart.min.js"></script>
	<script src="js/chart-data.js"></script>
	<script src="js/easypiechart.js"></script>
	<script src="js/easypiechart-data.js"></script>
	<script src="js/bootstrap-datepicker.js"></script>
	<script src="js/custom.js"></script>
	
</body>
</html>
